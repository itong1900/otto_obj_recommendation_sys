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

# ======= CONSTANTS ========
VALIDATION_RUN = False   ## Set to True when validating a new method, cut off sim matrix training in the 1st cycle
PARALLEL = 1024

## Weight of itemTotalLikes, and similarity matrices training
ACTION_WEIGHTS = np.array([1.0, 6.0, 3.0])  
MODES_TO_TRAIN = ["iuf", "time_decay"]
TRIM_CYCLES = 1000            ## trim full sim matrix every XX batches. 
SIM_MATRIX_TRIM_SIZE = 100   ## keep top XX similar items after a TRIM_CYCLE. 
RECALL_SIZE = 20             ## Not having reranking for this method, therefore 20.

## Weights for making inference
CLICKS_INF_WEGIHTS = np.array([3.0, 6.0, 3.0]) 
CARTS_INF_WEGIHTS = np.array([4.0, 2.0, 5.0])
ORDERS_INF_WEGIHTS = np.array([2.0, 6.0, 6.0])

def load_data():
    df = pd.read_csv("../../allData/validationData/train_meta_data.csv")
    df_test = pd.read_csv("../../allData/validationData/test_meta_data.csv")
    df = pd.concat([df, df_test]).reset_index(drop = True)
    npz = np.load("../../allData/validationData/train_core_data.npz")
    npz_test = np.load("../../allData/validationData/test_core_data.npz")
    aids = np.concatenate([npz['aids'], npz_test['aids']])
    ts = np.concatenate([npz['ts'], npz_test['ts']])
    ops = np.concatenate([npz['ops'], npz_test['ops']])

    df["start_idx"] = df['total_action'].cumsum().shift(1).fillna(0).astype(int)
    df["end_time"] = ts[df["start_idx"] + df["total_action"] - 1]

    return df, df_test, aids, ts, ops


def compute_sim_matrix(mode, item_total_likes, df, aids, ts, ops):
    """
    @param mode: "cosine" / "iuf" / "time_decay"
    @param item_total_likes: maps that contain popularity score of each aid
    @return sim matrix as requested mode and item_total_likes
    """

    fullSimMatrix = nb.typed.Dict.empty(
                key_type = nb.types.int64,
                value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
    max_idx = len(df)
    batch_idx = 1  ## compute sim matrix for PARALLEL # of rows per batch, have a total of max_idx/PARALLEL batches.
    for idx in tqdm(range(0, max_idx, PARALLEL)):
        rows = df.iloc[idx: min(idx + PARALLEL, max_idx)][['session', 'start_idx', 'total_action', 'session_start_time']].values
        itemCF_functions.getSimScoreBatch(aids, ts, ops, rows, fullSimMatrix, ACTION_WEIGHTS, item_total_likes, mode=mode)
        batch_idx += 1
        if batch_idx % TRIM_CYCLES == 0:
            print("batch_idx: ", batch_idx)
            itemCF_functions.trim_simMatrix_topk(fullSimMatrix, SIM_MATRIX_TRIM_SIZE)
            gc.collect()
            if VALIDATION_RUN:
                break
    
    ## in case number of batch is not divisable for TRIM_CYCLES
    itemCF_functions.trim_simMatrix_topk(fullSimMatrix, SIM_MATRIX_TRIM_SIZE)  
    ## max norm of each score
    itemCF_functions.maxNormSimMatrix(fullSimMatrix)
    
    return fullSimMatrix

# ==============================
# Section E: Main Logic in Making Inferences
# 1. clicks_inferences: time_decay sim matrix + regular action weights <1, 6, 3>.
# 2. carts_inferencs: iuf sim matrix + weights <4, 2, 5> (as clicks actions tend to lead to cart action next).
# 3. orders_inferences: iuf sim matrix + regular action weights <1, 6, 3>.
# ==============================
@nb.jit(nopython=True)
def inference_single_session(session, starting_idx, length, start_time, aids, ops, ts, result, full_sim_matrix, test_ops_weights):
    """
    Store top k candidates based on handcrafted rules.
    Final result will store as into result struct as result[session] = list_of_topK. 
    """
    
    ending_idx = starting_idx + length
    
    candidates = aids[starting_idx: ending_idx][::-1]
    candidates_ops = ops[starting_idx: ending_idx][::-1]
    
    ## record all potential aid that might be relevant
    potential_to_recommend = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)    

    ## get unique aid of each session 
    unique_aids = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
    for a in candidates:
        unique_aids[a] = 0
    
    if len(unique_aids) >= 20:   ## if the user has many actions, recommend based on the itemes it had been interacted with only.
        PREV_INTERACT_BONUS = 10
        sequence_weight = np.power(2, np.linspace(0.3, 1, len(candidates)))[::-1] - 1
        for aid, op, w in zip(candidates, candidates_ops, sequence_weight):
            if aid not in potential_to_recommend:
                potential_to_recommend[aid] = 0
            potential_to_recommend[aid] += w * test_ops_weights[op] * PREV_INTERACT_BONUS 
        #result[session] = np.array(heap_topk_return_list(potential_to_recommend, 20)) 
    else:  
        for idx in range(starting_idx, ending_idx):
            candidate = aids[idx]
            if candidate not in potential_to_recommend:
                potential_to_recommend[candidate] = np.inf ## ensure large weights on items had interacted with. 
    
    ## In case some items are duplicates, when potential_to_recommend not yet reach 20, impute with items from sim matrix
    if len(potential_to_recommend) < 80: ## CAUTIOUS: validation purpose only 
        sequence_weight = np.power(2, np.linspace(0.1, 1, len(candidates))) - 1   ## CHANGE_MADE: 0.3 -> 0.1
        for idx in range(starting_idx, ending_idx):
            candidate = aids[idx] 
            time_weight = 1 + 0.1 ** ((1662328791-ts[idx])/(1662328791-1659304800))    ## TODO: consider 
            candidate_realtime_weight = test_ops_weights[ops[idx]] * sequence_weight[idx-starting_idx] * time_weight  
            ## load the potential items to recommend,
            if candidate not in full_sim_matrix: 
                continue
            for similar_item in full_sim_matrix[candidate]:
#                 if similar_item in candidates:    ## skip the item if the it's already been interacted
#                     continue
                if similar_item not in potential_to_recommend:
                    potential_to_recommend[similar_item] = 0
                potential_to_recommend[similar_item] += full_sim_matrix[candidate][similar_item] * candidate_realtime_weight 
    
    result[session] = np.array(itemCF_functions.heap_topk_return_list(potential_to_recommend, RECALL_SIZE))   
    
@nb.jit(nopython=True)
def run_inference_parallel(rows, aids, ops, ts, result, full_sim_matrix, test_ops_weights):
    """
    Run inference single session in parallel by PARALLEL parameter. 
    """
    for row_idx in nb.prange(len(rows)):
        session, starting_idx, length, start_time = rows[row_idx]
        inference_single_session(session, starting_idx, length, start_time, aids, ops, ts, result, full_sim_matrix, test_ops_weights)


def storeItemTotalLike(aids, ops):
    """
    local method to return itemTotalLikes map.
    """
    item_total_likes = nb.typed.Dict.empty(
        key_type = nb.types.int64,
        value_type = nb.types.float64)

    itemCF_functions.getItemTotalLikesNaive(aids, ops, item_total_likes, ACTION_WEIGHTS)
    
    return item_total_likes


def store_result(df, df_test, aids, ops, ts, sim_matrix, inference_weight):
    """
    local method to store the result as maps. 
    {
        sessionXXX -> list_of_20,
        .....
        .....
    }
    """
    result = nb.typed.Dict.empty(
        key_type = nb.types.int64,
        value_type = nb.types.int64[:])

    for row_idx in tqdm(range(len(df) - len(df_test), len(df), PARALLEL)):
        start_row = row_idx
        end_row = min(row_idx + PARALLEL, len(df))
        rows = df.iloc[start_row: end_row][['session', 'start_idx', 'total_action', 'session_start_time']].values
        run_inference_parallel(rows, aids, ops, ts, result, sim_matrix, inference_weight)

    return result


def main(output_path):

    df, df_test, aids, ts, ops = load_data()
    
    ## Fit co-visitation matrices
    item_total_likes = storeItemTotalLike(aids, ops)

    simMatrices = {}   ## store a few different similarity matrices using different scoring system, for different prediction type

    for mode in MODES_TO_TRAIN:
        simMatrices[mode] = compute_sim_matrix(mode, item_total_likes, df, aids, ts, ops)
        gc.collect()

    ## make inference
    results_map = {}
    op_names = ["clicks", "carts", "orders"]
    infer_weights = [CLICKS_INF_WEGIHTS, CARTS_INF_WEGIHTS, ORDERS_INF_WEGIHTS]
    sim_matrix_types = ["time_decay", "iuf", "iuf"]

    for op_type, infer_weight, sim_matrix_type in zip(op_names, infer_weights, sim_matrix_types):
        results_map[op_type] = store_result(df, df_test, aids, ops, ts, simMatrices[sim_matrix_type], infer_weight)
    
    ## output the result
    subs = []
    op_names = ["clicks", "carts", "orders"]

    for result, op in zip([results_map["clicks"], results_map["carts"], results_map["orders"]], op_names):
        sub = pd.DataFrame({"session_type": result.keys(), "labels": result.values()})
        sub.session_type = sub.session_type.astype(str) + f"_{op}"
        sub.labels = sub.labels.apply(lambda x: " ".join(x.astype(str)))
        subs.append(sub)
        
    submission = pd.concat(subs).reset_index(drop=True)
    submission.to_csv(output_path + '/submission.csv', index = False)
    

if __name__ == "__main__":
    startTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=Path, required=True)
    args = parser.parse_args()
    main(args.output_path)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))   