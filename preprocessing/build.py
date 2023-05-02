import random

SAMPLE_FLAG = False


def rank_occurrence_cat(query, all_cat):
    cat_list = [(cat,) if type(cat) == str else cat for cat in all_cat]
    proc_query = query[:-1].lower()
    first_idx = None
    cat_num = len(cat_list)
    for q in proc_query.split():
        for idx in range(cat_num):
            for choice in cat_list[idx]:
                if choice == q:
                    if first_idx is not None:
                        return first_idx, idx
                    first_idx = idx
    return first_idx, None


def extract_single_feat_nl(cat_list, queries):
    best_occ = None
    random.shuffle(queries)
    for query in queries:
        occ = rank_occurrence_cat(query, cat_list)
        if occ[1] is not None:
            return occ, query
        if best_occ is None or occ[0] is not None:
            best_occ = occ
    return best_occ, query


def extract_single_feat_nl_once(cat_list, queries):
    random.shuffle(queries)
    occ = rank_occurrence_cat(queries[0], cat_list)
    return occ, queries[0]


def extract_feats_from_nl(data_cfg, nl, nl_view):
    total_nl = nl + nl_view
    total_nl_str = " ".join(total_nl).lower()
    
    feats = dict()
    nl_type, q_type = extract_single_feat_nl(data_cfg.TEXT_AUG_TYPE, total_nl)
    
    nl_color, q_color = extract_single_feat_nl(data_cfg.TEXT_AUG_COLOR, total_nl)
    
    
    if data_cfg.FEAT_ENG:
        nl_motion, q_motion = extract_single_feat_nl_once(data_cfg.TEXT_AUG_MOTION, nl)
        nl_size, q_size = extract_single_feat_nl_once(data_cfg.TEXT_AUG_SIZE, total_nl)
        for i in range(3):
            if nl_motion[0] is not None:
                q_motion1 = q_motion[:-1].lower().split()
                for mo_idx in range(len(q_motion1)):
                    if mo_idx < len(q_motion1) - 1 and q_motion1[mo_idx] == data_cfg.TEXT_AUG_MOTION[nl_motion[0]] \
                        and q_motion1[mo_idx + 1] == 'lane':
                        nl_motion, q_motion = extract_single_feat_nl_once(data_cfg.TEXT_AUG_MOTION, nl)
                        break
            else:
                break
        if nl_type[1] is None:
            nl_color = (nl_color[0], None)
    else:
        nl_motion, q_motion = extract_single_feat_nl(data_cfg.TEXT_AUG_MOTION, nl)
        nl_size, q_size = extract_single_feat_nl(data_cfg.TEXT_AUG_SIZE, total_nl)
    
    
    
    feats["intersection"] = int(data_cfg.TEXT_AUG_INTER in total_nl_str)
    
    feats["type1"] = len(data_cfg.TEXT_AUG_TYPE) if nl_type[0] is None else nl_type[0]
    feats["type2"] = len(data_cfg.TEXT_AUG_TYPE) if nl_type[1] is None else nl_type[1]
    feats["motion1"] = len(data_cfg.TEXT_AUG_MOTION) if nl_motion[0] is None else nl_motion[0]
    feats["motion2"] = len(data_cfg.TEXT_AUG_MOTION) if nl_motion[1] is None else nl_motion[1]
    feats["color1"] = len(data_cfg.TEXT_AUG_COLOR) if nl_color[0] is None else nl_color[0]
    feats["color2"] = len(data_cfg.TEXT_AUG_COLOR) if nl_color[1] is None else nl_color[1]
    feats["size1"] = len(data_cfg.TEXT_AUG_SIZE) if nl_size[0] is None else nl_size[0]
    feats["size2"] = len(data_cfg.TEXT_AUG_SIZE) if nl_size[1] is None else nl_size[1]
    
    all_q = [q_type, q_motion, q_color, q_size]
    
    feats["q"] = max(all_q,key=all_q.count)

    feats["aug_txt"] = ""
    if nl_type[0] is not None:
        aug_type = data_cfg.TEXT_AUG_TYPE[nl_type[0]]
        if type(aug_type) == str:
            feats["aug_txt"] = aug_type  + ". "
        else:
            feats["aug_txt"] = aug_type[0] + ". "
    
    if nl_color[0] is not None:
        aug_color = data_cfg.TEXT_AUG_COLOR[nl_color[0]]
        if type(aug_color) == str:
            feats["aug_txt"] = aug_color + " " + feats["aug_txt"]
        else:
            feats["aug_txt"] = aug_color[0]  + " " + feats["aug_txt"]

    if nl_size[0] is not None:
        aug_size = data_cfg.TEXT_AUG_SIZE[nl_size[0]]
        if type(aug_size) == str:
            feats["aug_txt"] = aug_size + " " + feats["aug_txt"]
        else:
            feats["aug_txt"] = aug_size[0]  + " " + feats["aug_txt"]

    feats["aug_car"] = ""
    if nl_motion[0] is not None:
        aug_motion = data_cfg.TEXT_AUG_MOTION[nl_motion[0]]
        if type(aug_motion) == str:
            feats["aug_car"] = aug_motion + ". "
        else:
            feats["aug_car"] = aug_motion  + ". "
    
    feats["aug_car_end"] = " intersection." if feats["intersection"] == 1 else ""
    
    
    return feats

