import gc
import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import time
from pathlib import Path
import argparse

import sys
sys.path.insert(0, '../item_collaborative_filter')
import itemCF_functions

from pure_itemCF_pipelines import load_data, store_item_total_like, compute_sim_matrix
from pure_itemCF_pipelines import PARALLEL, CLICKS_INF_WEGIHTS, CARTS_INF_WEGIHTS, ORDERS_INF_WEGIHTS

## This is script is very similar to pure_itemCF_piplines.py, but instead of using to directly generate 20 candidates,
## here it generates RECALL_SIZE_RERANK(currently set at 200) candiates, along with some feature engineering. 
## The output of this script will be a file with top 200 candidates of each session along with its features, 
## it will be used as LgbmRanker training in the next step. 

## A lot of functions or constants and same and therefore directly import from pure_itemCF_pipelines.

# ======= CONSTANTS ========
VALIDATION_RUN = False   ## Set to True when validating a new method, cut off sim matrix training in the 1st cycle
RECALL_SIZE_RERANK = 200

FEATURE_TUPLE_TEMPLATE = (bool(0), np.float64(0.0), np.float64(0.0), np.int64(0), np.int64(0), np.int64(0), np.float64(0.0), np.float64(0.0), 
                          np.int32(0), np.float32(0.0), np.float64(0.0), np.int32(0), np.float64(0.0), np.float64(0.0), np.float32(0.0), np.float32(0.0),
                          np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.int32(0), 
                          np.int32(0), np.int32(0), np.int32(0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), 
                          np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0))

## See README for detailed explanation of each feature. 
FEATURE_NAMES = ["prev_int", "seq_w_total", "time_w_total", "ops_w_total", "session_len", "num_uniuqe_aids", "CF_score", "itemTotalLike", 
                 "ref_time", "max_sim_score","mean_sim_score", "num_interact", "time_span", "action_recency", "seq_w_max", "seq_w_mean", 
                 "seq_w_min", "time_w_max", "time_w_mean", "time_w_min", "ops_w_max", "ops_w_mean", "ops_w_min", "num_clicks", 
                 "num_carts", "num_orders", "last_action_type", "time_to_now", "cf_incre_max", "cf_incre_mean", "cf_incre_min", "seqW_max_min_gap", 
                 "timeW_max_min_gap", "opsW_max_min_gap", "cf_incre_max_min_gap", "last_action_raw_seq", "raw_seq_sum", "raw_seq_max", "raw_seq_mean"]


@nb.jit(nopython=True)
def update_feature_vec(aid, features_tuple_arr, features_idx_map, new_feat_tuple):
    """
    This function dynamically update the features of each session-aid. 
    """
    ## append features
    if aid not in features_idx_map:
        features_tuple_arr.append(new_feat_tuple)
        new_pos = len(features_tuple_arr)-1
        ## save the position in the tuple arr
        features_idx_map[aid] = new_pos
    else: # <is_prev_int, seq_w, time_w, total_ops_w, session_len, # uniuqe aids, CF_score, aid's itemTotalLike >
        # ================== 8 ==
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot8_ref_time = features_tuple_arr[features_idx_map[aid]][8]
        else:
            slot8_ref_time = features_tuple_arr[features_idx_map[aid]][8] + new_feat_tuple[8]
        # ================== 9 ==
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot9_max_sim = 1
        else:
            slot9_max_sim = max(new_feat_tuple[9], features_tuple_arr[features_idx_map[aid]][9])
        # ================== 10 ==
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot10_mean_sim = 1
        else:
            slot10_mean_sim = ((features_tuple_arr[features_idx_map[aid]][10] * (features_tuple_arr[features_idx_map[aid]][8]-1) ) + new_feat_tuple[10] ) / features_tuple_arr[features_idx_map[aid]][8]
        # ================== 14 seq_w ==
        slot14_max_seq_w = max(new_feat_tuple[14], features_tuple_arr[features_idx_map[aid]][14])
        # ================== 15 ==
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot15_mean_seq_w = (features_tuple_arr[features_idx_map[aid]][1] + new_feat_tuple[1]) / (features_tuple_arr[features_idx_map[aid]][11] + new_feat_tuple[11])
        else:
            slot15_mean_seq_w = (features_tuple_arr[features_idx_map[aid]][1] + new_feat_tuple[1]) / (features_tuple_arr[features_idx_map[aid]][8] + new_feat_tuple[8])
        # ================== 16 ==
        slot16_min_seq_w = min(new_feat_tuple[16], features_tuple_arr[features_idx_map[aid]][16])
        # ================== 17 time_w == 
        slot17_max_time_w = max(new_feat_tuple[17], features_tuple_arr[features_idx_map[aid]][17])
        # ================== 18 == 
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot18_mean_time_w = (features_tuple_arr[features_idx_map[aid]][2] + new_feat_tuple[2]) / (features_tuple_arr[features_idx_map[aid]][11] + new_feat_tuple[11])
        else:
            slot18_mean_time_w = (features_tuple_arr[features_idx_map[aid]][2] + new_feat_tuple[2]) / (features_tuple_arr[features_idx_map[aid]][8] + new_feat_tuple[8])
        # ================= 19 ==
        slot19_min_time_w = min(new_feat_tuple[19], features_tuple_arr[features_idx_map[aid]][19])
        # ================= 20 ops_w ==
        slot20_max_ops_w = max(new_feat_tuple[20], features_tuple_arr[features_idx_map[aid]][20])
        # ================= 21 ==
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot21_mean_ops_w = (features_tuple_arr[features_idx_map[aid]][3] + new_feat_tuple[3]) / (features_tuple_arr[features_idx_map[aid]][11] + new_feat_tuple[11])
        else:
            slot21_mean_ops_w = (features_tuple_arr[features_idx_map[aid]][3] + new_feat_tuple[3]) / (features_tuple_arr[features_idx_map[aid]][8] + new_feat_tuple[8])
        # ================= 22 ==
        slot22_min_ops_w = min(new_feat_tuple[22], features_tuple_arr[features_idx_map[aid]][22]) #new_feat_tuple[22] if new_feat_tuple[22] < features_tuple_arr[features_idx_map[aid]][22] else features_tuple_arr[features_idx_map[aid]][22]
        # ================= 28 cf_incre ==
        slot28_max_cf_incre = max(new_feat_tuple[28], features_tuple_arr[features_idx_map[aid]][28])
        # ================= 29 ==
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot29_mean_cf_incre = new_feat_tuple[6] / (features_tuple_arr[features_idx_map[aid]][11] + new_feat_tuple[11])
        else:
            slot29_mean_cf_incre = new_feat_tuple[6] / (features_tuple_arr[features_idx_map[aid]][8] + new_feat_tuple[8])
        # ================= 30 ==
        slot30_min_cf_incre = min(new_feat_tuple[30], features_tuple_arr[features_idx_map[aid]][30])
        # ================= 37 raw_seq_max ==
        slot37_max_raw_seq = max(new_feat_tuple[37], features_tuple_arr[features_idx_map[aid]][37])
        # ================== 38 raw_seq_mean == 
        if features_tuple_arr[features_idx_map[aid]][0]: 
            slot38_mean_raw_seq = (features_tuple_arr[features_idx_map[aid]][36] + new_feat_tuple[36]) / (features_tuple_arr[features_idx_map[aid]][11] + new_feat_tuple[11])
        else:
            slot38_mean_raw_seq = (features_tuple_arr[features_idx_map[aid]][36] + new_feat_tuple[36]) / (features_tuple_arr[features_idx_map[aid]][8] + new_feat_tuple[8])
        

        features_tuple_arr[features_idx_map[aid]] = (new_feat_tuple[0], 
                                                     features_tuple_arr[features_idx_map[aid]][1] + new_feat_tuple[1], 
                                                     features_tuple_arr[features_idx_map[aid]][2] + new_feat_tuple[2], 
                                                     features_tuple_arr[features_idx_map[aid]][3] + new_feat_tuple[3],
                                                     new_feat_tuple[4],
                                                     new_feat_tuple[5],
                                                     new_feat_tuple[6],
                                                     new_feat_tuple[7],
                                                     slot8_ref_time,
                                                     slot9_max_sim,
                                                     slot10_mean_sim,
                                                     features_tuple_arr[features_idx_map[aid]][11] + new_feat_tuple[11],
                                                     new_feat_tuple[12],
                                                     features_tuple_arr[features_idx_map[aid]][13],  ## same reason as 26/27
                                                     slot14_max_seq_w,
                                                     slot15_mean_seq_w,
                                                     slot16_min_seq_w,
                                                     slot17_max_time_w,
                                                     slot18_mean_time_w,
                                                     slot19_min_time_w,
                                                     slot20_max_ops_w,
                                                     slot21_mean_ops_w,
                                                     slot22_min_ops_w,
                                                     features_tuple_arr[features_idx_map[aid]][23] + new_feat_tuple[23], 
                                                     features_tuple_arr[features_idx_map[aid]][24] + new_feat_tuple[24],
                                                     features_tuple_arr[features_idx_map[aid]][25] + new_feat_tuple[25],
                                                     features_tuple_arr[features_idx_map[aid]][26],  ## first time hit is the exact value to look at
                                                     features_tuple_arr[features_idx_map[aid]][27],  ## same reason as above
                                                     slot28_max_cf_incre,
                                                     slot29_mean_cf_incre,
                                                     slot30_min_cf_incre,
                                                     slot14_max_seq_w - slot16_min_seq_w,
                                                     slot17_max_time_w - slot19_min_time_w,
                                                     slot20_max_ops_w - slot22_min_ops_w,
                                                     slot28_max_cf_incre - slot30_min_cf_incre,
                                                     features_tuple_arr[features_idx_map[aid]][35],
                                                     features_tuple_arr[features_idx_map[aid]][36] + new_feat_tuple[36], 
                                                     slot37_max_raw_seq,
                                                     slot38_mean_raw_seq
                                                     )


@nb.jit(nopython=True)
def save_feature_single_session(session, starting_idx, length, start_time, aids, ops, ts, result, full_sim_matrix, item_total_likes, test_ops_weights):
    NOW_TIME = ts[-1] ## ts of latest avaiable action
    PREV_INTERACT_BONUS = 30
    NEARBY_ACTION_BONUS = 1.5
    
    ending_idx = starting_idx + length 
    end_time = ts[ending_idx - 1]
    time_span = end_time - start_time
    
    candidates = aids[starting_idx: ending_idx][::-1]
    candidates_ops = ops[starting_idx: ending_idx][::-1]
    
    ## record all potential aid that might be relevant
    potential_to_recommend = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)
    
    ## get unique aid of each session 
    unique_aids = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
    for a in candidates:
        unique_aids[a] = 0
    
    ## Sequence weight to all the candidates, from near to far 
    sequence_weight = np.power(2, np.linspace(0.3, 1, len(candidates)))[::-1] - 1
    
    raw_sequence = np.arange(1, len(candidates) + 1)
    
    ## Time weight of all candidates, from near to far
    time_weights = []
    time_lapse = end_time - start_time + 1  ## +1 to avoid zero
    for idx in range(starting_idx, ending_idx):
        if end_time - ts[idx] < 2 * 60 * 60:   ## apply nearby action bonus
            time_weight = (1 + 0.5 ** ((end_time - ts[idx])/time_lapse)) * NEARBY_ACTION_BONUS
        else:
            time_weight = 1 + 0.5 ** ((end_time - ts[idx])/time_lapse)
        time_weights.append(time_weight)
    time_weights = time_weights[::-1]
    
    ## feature vector template: [aid: <is_prev_int, seq_w, time_w, associated_action, session_len,.. >]
    features_tuple_arr = nb.typed.List()
    features_tuple_arr.append(FEATURE_TUPLE_TEMPLATE)
    features_idx_map = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)

    helper_idx = ending_idx - 1
    ## making inference
    if len(unique_aids) >= 20:  
        for aid, op, seq_w, time_w, raw_seq in zip(candidates, candidates_ops, sequence_weight, time_weights, raw_sequence):
            if aid not in potential_to_recommend:
                potential_to_recommend[aid] = 0
            ## caculate scores
            cf_incre = seq_w * time_w * test_ops_weights[op]
            potential_to_recommend[aid] += cf_incre #* PREV_INTERACT_BONUS
            ## append features
            update_feature_vec(
                aid, features_tuple_arr, features_idx_map, \
                    (1, seq_w, time_w, test_ops_weights[op], length, len(unique_aids), potential_to_recommend[aid], item_total_likes[aid], 
                     100, 1, 1, 1, time_span, end_time-ts[helper_idx], seq_w, seq_w,
                     seq_w, time_w, time_w, time_w, test_ops_weights[op], test_ops_weights[op], test_ops_weights[op], op==0, 
                     op==1, op==2, op, NOW_TIME-ts[helper_idx], cf_incre, cf_incre, cf_incre, 0, 
                     0, 0, 0, raw_seq, raw_seq, raw_seq, raw_seq)
                              )
            helper_idx -= 1
    else:   ## otherwise, fill the rest with similar items.
        for aid, op, seq_w, time_w, raw_seq in zip(candidates, candidates_ops, sequence_weight, time_weights, raw_sequence):
            if aid not in potential_to_recommend:
                potential_to_recommend[aid] = 0
            ## get the scores
            cf_incre = seq_w * time_w * test_ops_weights[op] * PREV_INTERACT_BONUS
            potential_to_recommend[aid] += cf_incre
            ## append features
            update_feature_vec(aid, features_tuple_arr, features_idx_map, \
                (1, seq_w, time_w, test_ops_weights[op], length, len(unique_aids), potential_to_recommend[aid], \
                    item_total_likes[aid], 100, 1, 1, 1, time_span, end_time-ts[helper_idx], seq_w, seq_w, seq_w, \
                        time_w, time_w, time_w, test_ops_weights[op], test_ops_weights[op], test_ops_weights[op], op==0, op==1, op==2, op, NOW_TIME - ts[helper_idx],\
                            cf_incre, cf_incre, cf_incre, 0, 0, 0, 0, raw_seq, raw_seq, raw_seq, raw_seq))
            ## adding the similar items, if full_sim_matrix don't have such record, skip. 
            if aid not in full_sim_matrix:
                continue
            for similar_item in full_sim_matrix[aid]:
                ## if sim_item is in candidates, would be included above anyways, skip 
                if similar_item in candidates:
                    continue
                if similar_item not in potential_to_recommend:
                    potential_to_recommend[similar_item] = 0
                
                cf_incre = seq_w * time_w * test_ops_weights[op] * full_sim_matrix[aid][similar_item]
                potential_to_recommend[similar_item] += cf_incre  ## no PREV_INTERACT_BONUS as expected, replaced with sim_matrix scores
                ## append features
                update_feature_vec(similar_item, features_tuple_arr, features_idx_map, \
                    (0, seq_w, time_w, test_ops_weights[op], length, len(unique_aids), potential_to_recommend[similar_item], \
                        item_total_likes[similar_item], 1, full_sim_matrix[aid][similar_item], full_sim_matrix[aid][similar_item], 0, \
                            time_span, end_time-ts[ending_idx-1], seq_w, seq_w, seq_w, time_w, time_w, time_w, test_ops_weights[op], test_ops_weights[op], test_ops_weights[op], op==0, op==1, op==2, op,\
                                NOW_TIME-ts[helper_idx], cf_incre, cf_incre, cf_incre, 0, 0, 0, 0, raw_seq, raw_seq, raw_seq, raw_seq))
            helper_idx -= 1

    result[session] = np.array(itemCF_functions.heap_topk_return_list(potential_to_recommend, RECALL_SIZE_RERANK))  ## Set number to recall. 
    
    feature_tuples_this_session = []
    for aid in result[session]:
#         features_save[(session, aid)] = features_tuple_arr[features_idx_map[aid]]
        feature_tuples_this_session.append(features_tuple_arr[features_idx_map[aid]])
    
    return feature_tuples_this_session


def load_gt_tables(type):
    """ type -> carts / orders """
    gt_labels = pd.read_json("/kaggle/input/localvalidationlabels/test_labels.jsonl", lines=True)
    gt_labels['aids'] = gt_labels["labels"].apply(lambda x: x.get(type))
    gt_labels = gt_labels[gt_labels.aids.notnull()]
    gt_labels = gt_labels.drop("labels", axis = 1)
    ## ========= special df to identify the unique session id to look at ================
    valid_gt_sessions = gt_labels.drop("aids", axis = 1) 
    ## ========================================================================
    ## keep go on for gt labels processing
    gt_labels = gt_labels.set_index(['session']).apply(pd.Series.explode).reset_index()
    gt_labels["gt"] = 1
    return valid_gt_sessions, gt_labels

def process_batch_pipeline(rawDf, valid_gt_sessions, gt_labels):
    """ rawDf -> Df with session, aids(100), feature_tuple """
    ## join valid_gt_session with rawDf, now only gt_features in valid sessions(have at least 1 aid to predict) are kept
    gt_features_valid_session = pd.merge(rawDf, valid_gt_sessions, on="session")

    ## Now explode the whole valid_gt_session aids, these session - aid are served as the train/val/test data for the reranker model, 
    ## for orders, 
    ## for carts, a total of 569697 correct guesses(not 100% included in the recall)
    gt_features_valid_session = gt_features_valid_session.set_index(['session']).apply(pd.Series.explode).reset_index()

    ## finally, attach the gt_lables 1/null to the df to return
    final_df = pd.merge(gt_features_valid_session, gt_labels, on=["session", "aids"], how='left')

    ## open up the feature tuple 
    features = np.vstack(final_df["feature_tuple"].values)
    temp_df = pd.DataFrame(features)
    del features
    temp_df.columns = [f'{feat_name}' for feat_name in FEATURE_NAMES]
    final_df[temp_df.columns] = temp_df
    del temp_df

    final_df = final_df.drop("feature_tuple", axis = 1)

    return final_df


def main(output_path, action_type):

    df, df_test, aids, ts, ops = load_data()
    
    ## Fit co-visitation matrices
    item_total_likes = store_item_total_like(aids, ops)

    if action_type == "clicks":  ## NOTE: Reduce recall size when doing clicks, as data size much larger. 
        sim_matrix = compute_sim_matrix("time_decay", item_total_likes, df, aids, ts, ops)
        inference_weight = CLICKS_INF_WEGIHTS
    elif action_type == "carts":
        sim_matrix = compute_sim_matrix("iuf", item_total_likes, df, aids, ts, ops)
        inference_weight = CARTS_INF_WEGIHTS
    elif action_type == "orders":
        sim_matrix = compute_sim_matrix("iuf", item_total_likes, df, aids, ts, ops)
        inference_weight = ORDERS_INF_WEGIHTS
    else:
        print("Wrong type input")

    gc.collect()

    ## Store featuers
    result = nb.typed.Dict.empty(
        key_type = nb.types.int64,
        value_type = nb.types.int64[:])

    features_all_sessions = [] # session, aid, feature tuple

    ## Given there are 1671803 sessions in total, we separate them into K batches
    K = 40
    session_per_batch = len(df_test) // K 
    row_idx_cutoffs = [(len(df) - len(df_test)) + (PARALLEL * (session_per_batch//PARALLEL) ) * i for i in range(1, K+3)]   ## batch process every 1024 * 136 rows
    feature_batch_id = 0

    valid_gt_sessions, gt_labels = load_gt_tables(action_type)
    print("finish loading the gt datas")

    for row_idx in tqdm(range(len(df) - len(df_test), len(df), PARALLEL)):
        start_row = row_idx
        end_row = min(row_idx + PARALLEL, len(df))
        rows = df.iloc[start_row: end_row][['session', 'start_idx', 'total_action', 'session_start_time']].values  
        ## run things in parallel
        for row_idx in nb.prange(len(rows)):
            session, starting_idx, length, start_time = rows[row_idx]
            features_tuples_this_session = save_feature_single_session(session, starting_idx, length, start_time, aids, ops, ts, result, sim_matrix, item_total_likes, inference_weight)
            features_all_sessions.append(features_tuples_this_session)
        
        if (start_row in row_idx_cutoffs) or (end_row == len(df)):
            ## save batch result
            rawDf = pd.DataFrame({"session": result.keys(), "aids": result.values(), "feature_tuple": features_all_sessions})
            batch_result = process_batch_pipeline(rawDf, valid_gt_sessions, gt_labels)
            batch_result.to_parquet(f"{output_path}/{action_type}_batch_result_{feature_batch_id}.parquet")
            ## clean the memory for next batch
            del batch_result, rawDf, features_all_sessions, result
            gc.collect()
            ## progress update
            print(f"feature_batch_{feature_batch_id} completes saving.")
            feature_batch_id += 1
            ## initiate the struct for new batch again
            result = nb.typed.Dict.empty(
                key_type = nb.types.int64,
                value_type = nb.types.int64[:])
            features_all_sessions = []


if __name__ == "__main__":
    startTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--action-type', required=True)
    args = parser.parse_args()
    main(args.output_path, args.action_type, args.have_gt)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))   