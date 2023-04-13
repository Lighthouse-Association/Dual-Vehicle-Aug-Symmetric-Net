import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import wandb

from .benchmark import Timer, get_mrr, accuracy
from .utils import results_record
from .postprocessing import rerank_params_grid_search
from .benchmark import AverageMeter, ProgressMeter
from .preprocessing import extract_feats_from_nl


best_mrr_eval = 0.
best_mrr_eval_by_test = 0.


def evaluate_by_test_all(model, valloader, epoch, cfg, index=-1, args=None, tokenizer=None, optimizer=None):
    """ evaluate crop, motion and merge features"""
    if not valloader:
        return
    global best_mrr_eval_by_test
    print(f"====> Test::::{valloader.dataset.name}")
    evl_timer = Timer()
    model.eval()

    feat_num = 4
    all_visual_embeds = [dict() for _ in range(feat_num)]
    out = [dict() for _ in range(feat_num)]
    with torch.no_grad():
        for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader)):
            vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda())
            for i in range(len(track_id)): # bs
                for feat_idx in range(feat_num):
                    vis_embed = vis_embed_list[feat_idx]
                    if track_id[i] not in out:
                        out[feat_idx][track_id[i]] = dict()
                    out[feat_idx][track_id[i]][frames_id[i].item()] = vis_embed[i, :]
        for track_id in out[-1].keys():
            for feat_idx in range(feat_num):
                img_feat = out[feat_idx][track_id]
                tmp = []
                for fid in img_feat:
                    tmp.append(img_feat[fid])
                tmp = torch.stack(tmp)
                tmp = torch.mean(tmp, 0)
                all_visual_embeds[feat_idx][track_id] = tmp

    all_lang_embeds = [dict() for _ in range(feat_num)]
    with open(cfg.DATA.EVAL_JSON_PATH) as f:
        print(f"====> Query {cfg.DATA.EVAL_JSON_PATH} load")
        queries = json.load(f)
    with torch.no_grad():
        for q_id in tqdm(queries.keys()):
            text = queries[q_id]['nl'][:-1]
            car_text = queries[q_id]['nl'][-1:]
            if cfg.DATA.USE_FEATS_QUERY:
                l_feats = extract_feats_from_nl(cfg.DATA, queries[q_id]['nl'], queries[q_id]['nl_other_views'])
                text = l_feats["q"]
                car_text = text
            if cfg.DATA.USE_FEATS_AUG_TEXT:
                lang_feats_dict = extract_feats_from_nl(cfg.DATA, queries[q_id]['nl'], queries[q_id]['nl_other_views'])
                text = [lang_feats_dict["aug_txt"] + tt for tt in text]
                car_text = [lang_feats_dict["aug_car"] + ctxt + lang_feats_dict["aug_car_end"] for ctxt in car_text]

            # same dual Text
            if cfg.MODEL.SAME_TEXT:
                car_text = text

            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            if 'dual-text' in cfg.MODEL.NAME:
                car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                            car_tokens['input_ids'].cuda(),
                                                            car_tokens['attention_mask'].cuda())
            else:
                lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
            for feat_idx in range(feat_num):
                lang_embeds = lang_embeds_list[feat_idx]
                all_lang_embeds[feat_idx][q_id] = lang_embeds

    all_sim = [list() for _ in range(feat_num)]
    with torch.no_grad():
        visual_embeds = [list() for _ in range(feat_num)]
        for q_id in all_visual_embeds[-1].keys():
            for feat_idx in range(feat_num):
                visual_embeds[feat_idx].append(all_visual_embeds[feat_idx][q_id])
        visual_embeds = [torch.stack(embeds) for embeds in visual_embeds] # 3x
        for q_id in tqdm(all_visual_embeds[-1].keys()):
            for feat_idx in range(feat_num):
                lang_embeds = all_lang_embeds[feat_idx][q_id]
                cur_sim = torch.mean(torch.matmul(lang_embeds, visual_embeds[feat_idx].T), 0, keepdim=True)
                all_sim[feat_idx].append(cur_sim)

    all_sim = [torch.cat(sim) for sim in all_sim]

    def compute_and_record(sim, name):
        sim_t_2_i = sim
        sim_i_2_t = sim_t_2_i.t()

        loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda())
        loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(sim_t_2_i.size(0)).cuda())
        loss = (loss_t_2_i + loss_i_2_t) / 2

        acc5, acc10 = accuracy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda(), topk=(5, 10))
        mrr_ = get_mrr(sim_t_2_i)
        all_mrr = mrr_.item() * 100
        results_record(name, valloader.dataset.name, epoch, loss.item(),
                       all_mrr, acc5[0], acc10[0], is_test=True)
        wandb.log({
            "val_mrr": all_mrr,
            "val_acc@10": acc10[0],
            "val_acc@5": acc5[0]
        })
        return all_mrr

    sum_sim = 0.
    for idx, sim in enumerate(all_sim):
        sum_sim += sim
        compute_and_record(sim, args.logs_dir.split('/')[-1] + f'feat_{idx}')
    sum_mrr = compute_and_record(sum_sim, args.logs_dir.split('/')[-1] + 'feat_all')

    if args.eval_only and cfg.TEST.RERANK:
        rerank_mrr, rerank_params, rerank_sim = rerank_params_grid_search(all_lang_embeds[-1], all_visual_embeds[-1])
        print(f"====> grid search rerank, best mrr = {rerank_mrr}, params: k1={rerank_params[0]}, k2={rerank_params[1]}, eps={rerank_params[2]}")

    print(f'Epoch {epoch} running time: ', evl_timer)
    print(f'Logs dir: {args.logs_dir} ')

    if args.eval_only:
        return
    if sum_mrr > best_mrr_eval_by_test:
        # save time
        best_mrr_eval_by_test = sum_mrr
        checkpoint_file = args.logs_dir + "/checkpoint_best_eval_all.pth"
        args.ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)

def evaluate_by_test(model, valloader, epoch, cfg, index=-1, args=None, tokenizer=None, optimizer=None):
    """ evaluate merge features"""
    if not valloader:
        return
    global best_mrr_eval_by_test
    print(f"====> Test::::{valloader.dataset.name}")
    evl_timer = Timer()
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Clip Loss', ':.4e')
    mrr = AverageMeter('MRR', ':6.4f')
    top10_acc = AverageMeter('Acc@10', ':6.4f')
    top5_acc = AverageMeter('Acc@5', ':6.4f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time, losses, mrr, top10_acc, top5_acc],
        prefix="Test Epoch: [{}]".format(epoch))
    epo_timer = Timer()

    all_visual_embeds = dict()
    out = dict()
    with torch.no_grad():
        for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(valloader)):
            vis_embed_list = model.module.encode_images(image.cuda(), motion.cuda())
            vis_embed = vis_embed_list[index]
            for i in range(len(track_id)):
                if track_id[i] not in out:
                    out[track_id[i]] = dict()
                out[track_id[i]][frames_id[i].item()] = vis_embed[i, :]
        for track_id, img_feat in out.items():
            tmp = []
            for fid in img_feat:
                tmp.append(img_feat[fid])
            tmp = torch.stack(tmp)
            tmp = torch.mean(tmp, 0)
            all_visual_embeds[track_id] = tmp


    all_lang_embeds = dict()
    with open(cfg.DATA.EVAL_JSON_PATH) as f:
        print(f"====> Query {cfg.DATA.EVAL_JSON_PATH} load")
        queries = json.load(f)
    with torch.no_grad():
        for q_id in tqdm(all_visual_embeds.keys()):
            text = queries[q_id]['nl'][:-1]
            car_text = queries[q_id]['nl'][-1:]
            if cfg.DATA.USE_FEATS_QUERY:
                l_feats = extract_feats_from_nl(cfg.DATA, queries[q_id]['nl'], queries[q_id]['nl_other_views'])
                text = l_feats["q"]
                car_text = text
            if cfg.DATA.USE_FEATS_AUG_TEXT:
                lang_feats_dict = extract_feats_from_nl(cfg.DATA, queries[q_id]['nl'], queries[q_id]['nl_other_views'])
                text = [lang_feats_dict["aug_txt"] + tt for tt in text]
                car_text = [lang_feats_dict["aug_car"] + ctxt + lang_feats_dict["aug_car_end"] for ctxt in car_text]

            # same dual Text
            if cfg.MODEL.SAME_TEXT:
                car_text = text

            tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
            if 'dual-text' in cfg.MODEL.NAME:
                car_tokens = tokenizer.batch_encode_plus(car_text, padding='longest', return_tensors='pt')
                lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                            car_tokens['input_ids'].cuda(),
                                                            car_tokens['attention_mask'].cuda())
            else:
                lang_embeds_list = model.module.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
            lang_embeds = lang_embeds_list[index]
            all_lang_embeds[q_id] = lang_embeds

    all_sim = []
    with torch.no_grad():
        visual_embeds = []
        for q_id in all_visual_embeds.keys():
            visual_embeds.append(all_visual_embeds[q_id])
        visual_embeds = torch.stack(visual_embeds)
        for q_id in tqdm(all_visual_embeds.keys()):
            lang_embeds = all_lang_embeds[q_id]
            cur_sim = torch.mean(torch.matmul(lang_embeds, visual_embeds.T), 0, keepdim=True)
            all_sim.append(cur_sim)

    if args.eval_only and cfg.TEST.RERANK:
        rerank_mrr, rerank_params, rerank_sim = rerank_params_grid_search(all_lang_embeds, all_visual_embeds)
        print(f"====> grid search rerank, best mrr = {rerank_mrr}, params: k1={rerank_params[0]}, k2={rerank_params[1]}, eps={rerank_params[2]}")

    all_sim = torch.cat(all_sim)
    sim_t_2_i = all_sim
    sim_i_2_t = sim_t_2_i.t()

    loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(sim_t_2_i.size(0)).cuda())
    loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(sim_t_2_i.size(0)).cuda())
    loss = (loss_t_2_i + loss_i_2_t) / 2

    acc5, acc10 = accuracy(all_sim, torch.arange(all_sim.size(0)).cuda(), topk=(5, 10))
    mrr_ = get_mrr(all_sim)
    all_mrr = mrr_.item() * 100

    losses.update(loss.item(), image.size(0))
    mrr.update(all_mrr, image.size(0))
    top10_acc.update(acc10[0], image.size(0))
    top5_acc.update(acc5[0], image.size(0))
    batch_time.update(epo_timer.update())
    
    wandb.log({
        "val_mrr": all_mrr,
        "val_acc@10": acc10[0],
        "val_acc@5": acc5[0]
    })

    progress.display(batch_idx)

    print(f'Epoch {epoch} running time: ', evl_timer)
    print(f'Logs dir: {args.logs_dir} ')

    results_record(args.logs_dir.split('/')[-1], valloader.dataset.name, epoch, loss.item(), all_mrr, acc10[0], acc5[0], is_test=True)
    if args.eval_only:
        return
    if all_mrr > best_mrr_eval_by_test:
        # save time
        best_mrr_eval_by_test = all_mrr
        checkpoint_file = args.logs_dir + "/checkpoint_best_eval.pth"
        args.ossSaver.save_ckpt(checkpoint_file, epoch, model, optimizer)
