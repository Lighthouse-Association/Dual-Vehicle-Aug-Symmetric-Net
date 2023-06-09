import json
import torch
import torch.multiprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader
import os.path as osp
from smart_open import open

from .configs import set_experiment
from .models import build_model
from .dataloaders.local_aggr import CityFlowNLInferenceDatasetLAggr
from .preprocessing import extract_feats_from_nl
import torchvision
from transformers import BertTokenizer,RobertaTokenizer

SAMPLE_FLAG = False

def inference_vis_and_lang(config_name, args, enforced=False):
    arch_name = "local_aggr"
    cfg = set_experiment(arch_name, config_name)

    checkpoint_name = cfg.TEST.RESTORE_FROM.split('/')[-1].split('.')[0]
    save_dir = 'extracted_feats/' + config_name

    feat_pth_path = save_dir + '/img_lang_feat_%s.pth' % checkpoint_name
    feat_pth_path = args.ossSaver.get_s3_path(feat_pth_path)
    try:
        with open(feat_pth_path):
            return feat_pth_path
    except Exception as e:
        pass
    

    # if args.ossSaver.check_s3_path(feat_pth_path):
    #     if not enforced and refile.s3_isfile(feat_pth_path):
    #         return feat_pth_path
    # else:
    #     if not enforced and osp.isfile(feat_pth_path):
    #         return feat_pth_path

    print(f"====> Generating {feat_pth_path}")

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_data = CityFlowNLInferenceDatasetLAggr(cfg.DATA, transform=transform_test)
    testloader = DataLoader(dataset=test_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, )

    args.resume = True
    args.use_cuda = True
    cfg.MODEL.NUM_CLASS = 2155

    model = build_model(cfg, args)

    if cfg.MODEL.BERT_TYPE == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
    else:
        assert False

    model.eval()

    index = cfg.MODEL.MAIN_FEAT_IDX

    all_lang_embeds = dict()
    with open(cfg.TEST.QUERY_JSON_PATH) as f:
        print(f"====> Query {cfg.TEST.QUERY_JSON_PATH} load")
        queries = json.load(f)
    with torch.no_grad():
        for text_id in tqdm(queries):
            text = queries[text_id]['nl'][:-1]
            car_text = queries[text_id]['nl'][-1:]
            
            if cfg.DATA.USE_FEATS_QUERY:
                l_feats = extract_feats_from_nl(cfg.DATA, queries[text_id]['nl'], queries[text_id]['nl_other_views'])
                text = l_feats["q"]
                car_text = text
            
            if cfg.DATA.USE_FEATS_AUG_TEXT:
                lang_feats_dict = extract_feats_from_nl(cfg.DATA, queries[text_id]['nl'], queries[text_id]['nl_other_views'])
                text = [lang_feats_dict["aug_txt"] + tt for tt in text]
                car_text = [lang_feats_dict["aug_car"] + ctxt + lang_feats_dict["aug_car_end"] for ctxt in car_text]
                global SAMPLE_FLAG
                if not SAMPLE_FLAG:
                    print("Sample text:", text[0])
                    SAMPLE_FLAG = True

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
            all_lang_embeds[text_id] = lang_embeds.data.cpu().numpy()

    all_visual_embeds = dict()
    out = dict()
    with torch.no_grad():
        for batch_idx, (image, motion, track_id, frames_id) in tqdm(enumerate(testloader)):
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
            all_visual_embeds[track_id] = tmp.data.cpu().numpy()



    feats = (all_visual_embeds, all_lang_embeds)

    args.ossSaver.save_pth(feat_pth_path, feats)

    return feat_pth_path


# def main():
#     args, cfg = prepare_start()

#     config_dict = {
#         "single_baseline_aug1_plus": 1.,
#     }

#     config_file_list = list(config_dict.keys())
#     merge_weights = list(config_dict.values())

#     for config_name in config_file_list:
#         vis_pkl, lang_pkl = inference_vis_and_lang(config_name, args, enforced=False)


# if __name__ == '__main__':
#     main()
