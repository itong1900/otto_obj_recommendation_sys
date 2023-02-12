import heapq
import numba as nb
import numpy as np
import math

# ======= CONSTANTS ========
## will look up to 200 items prior to construct the co-visitation matrix. 
LOOKBACK_WINDOW = 200
## Consider no similarity if two interactions are happend more than TIME_FILTER_WINDOW away from each other.
TIME_FILTER_WINDOW = 1.2   ## hour

# =====================================
# Section A: Utils Functions 
# 1. Count Item Total likes: The similary score will be normalized by "Item Total Like Scores". In theory, popular items should have less weight in simiarity score.
# 2. Trimming function: Helpful managing memoery usage. 
# 3. Method for normalization: Mostly item total like normalization, and max norm(make all sim score between 0 and 1) of the score. 
# ======================================

# Methods for counting Item Total Likes
@nb.jit(nopython=True)
def getItemTotalLikesNaive(aids, ops, item_total_likes, action_weights):
    """
    Stores the total like score of itemXXX in item_total_likes, based on action_weights parameter. np.array([X, Y, Z])
    """
    for idx, item in enumerate(aids):
        if item not in item_total_likes: 
            item_total_likes[item] = 0
        item_total_likes[item] += action_weights[ops[idx]]   ## TODO: For time decay, consider replace with 1, for iuf keep this. 

# Methods for rank and trim the sim score dict
@nb.jit(nopython = True)
def heap_topk(item_cnt_dict, cap):
    """
    get the top cap(k) elements of the cnt dict based on value, using a min-heap structure
    """
    dic = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
    q = [(np.float64(0), np.int64(0)) for _ in range(0)]  ## generate empty queue to implement a heap, 
    for item_ref, sim_score in item_cnt_dict.items():   ## read in the dict in heap structure
        heapq.heappush(q, (sim_score, item_ref))   ## push the <sim_score, item_ref_id> pair into min-heap, using sim_score for order
        if len(q) > cap:
            heapq.heappop(q)
            
    res = [heapq.heappop(q) for _ in range(len(q))][::-1]
    for i in range(len(res)):
        dic[res[i][1]] = res[i][0]
    
    return dic
   
@nb.jit(nopython = True)
def trim_simMatrix_topk(fullSimMatrix, k):
    """
    trim top k items of each "itemX: {itemY: score1, ...}" pair in fullSimMatrix based on sim scores. 
    """
    for item, item_cnt_dict in fullSimMatrix.items():
        fullSimMatrix[item] = heap_topk(item_cnt_dict, k)

# Methods for score normalization
@nb.jit(nopython=True)
def maxNormSimMatrix(fullSimMatrix):
    for aid_1, relations in fullSimMatrix.items():
        max_num = -np.inf
        for _, sim_score in relations.items():
            if sim_score > max_num:
                max_num = sim_score
        for aid_2, sim_score in relations.items():
            fullSimMatrix[aid_1][aid_2] = sim_score / max_num


# =================================
# Section B: Sim Score Computation functions
# =================================
@nb.jit(nopython=True)
def getSimScoresSingleRow(pairs_this_row, start_time, start_idx, length, aids, ts, ops, item_total_likes, action_weights, mode):
    """
    Get the sim scores of items within single session, can be ran in parallel within each batch. 
    """
    max_idx = start_idx + length
    min_idx = max(max_idx - LOOKBACK_WINDOW, start_idx)  
    for i in range(min_idx, max_idx):
        for j in range(i+1, max_idx):
            if ts[j] - ts[i] > TIME_FILTER_WINDOW * 60 * 60: continue
            if aids[i] == aids[j]: continue
            
            if mode == "cosine":
                w_ij = action_weights[ops[j]] 
                w_ji = action_weights[ops[i]] 

            elif mode == "iuf":  ## penalize users that had lots of actions TODO: consider location weight
                loc_weight = 0.5**(abs(i-j))   #math.exp(-0.02 * abs(i-j)) 
                time_gap_weight = 0.5 ** (abs(ts[i]-ts[j]) / (1.5*60*60))  
                w_ij = action_weights[ops[j]] * time_gap_weight * loc_weight / math.log1p(length)
                w_ji = action_weights[ops[i]] * time_gap_weight * loc_weight / math.log1p(length)
            elif mode == "time_decay":
                ## calculate some time weights of each item, more weights are given when ts is later. #TODO: try adding (i-j) location weight, exponential weight, 0.5 ** (abs(i-j + 1)), 
                loc_weight = 0.5**(abs(i-j))   #math.exp(-0.02 * abs(i-j)) 
                #time_i = 1 + 0.1 ** ((1662328791-ts[i])/(1662328791-1659304800)) #1 + 3 * (ts[i] + start_time - 1659304800) / (1662328791 - 1659304800) #  #(1 - 0.8 *(TEST_END_TS - ts[i]) / TIME_SPAN) ** 0.5 # 0.2~1 #   ## time decay weight for item i 
                #time_j = 1 + 0.1 ** ((1662328791-ts[j])/(1662328791-1659304800))  # 1 + 3 * (ts[j] + start_time - 1659304800) / (1662328791 - 1659304800) # #  #(1 - 0.8 *(TEST_END_TS - ts[j]) / TIME_SPAN) ** 0.5   # 
                time_i = 1 + 1/(1 + math.exp(10*( ((1662328791-ts[i])/(1662328791-1659304800)) - 0.6  )))
                time_j = 1 + 1/(1 + math.exp(10*( ((1662328791-ts[j])/(1662328791-1659304800)) - 0.6  )))
                
                time_gap_weight = 0.5 ** (abs(ts[i]-ts[j]) / (1.5*60*60))  
                
                w_ij = action_weights[ops[j]] * loc_weight * time_gap_weight * time_i / math.log1p(length)
                w_ji = action_weights[ops[i]] * loc_weight * time_gap_weight * time_j / math.log1p(length)
                
            pairs_this_row[(aids[i], aids[j])] = w_ij / (item_total_likes[aids[i]] * item_total_likes[aids[j]]) ** 0.1
            pairs_this_row[(aids[j], aids[i])] = w_ji / (item_total_likes[aids[i]] * item_total_likes[aids[j]]) ** 0.1

@nb.jit(nopython=True, parallel=True, cache=True)
def getSimScoreBatch(aids, ts, ops, rows, fullSimMatrix, action_weights, item_total_likes, mode):
    nrows = len(rows)
    pairs_this_batch = [{(0, 0): 0.0 for _ in range(0)} for _ in range(nrows)]
    ## get the sim scores of each batch in seperate sub dict in pairs_this_batch
    for row_i in nb.prange(nrows):  ## run each row of the batch in parallel
        _, start_idx, length, start_time = rows[row_i]
        getSimScoresSingleRow(pairs_this_batch[row_i], start_time, start_idx, length, aids, ts, ops, item_total_likes, action_weights, mode)
    ## merge pairs_this_batch into the fullSimMatrix
    for row_i in range(nrows):
        for (aid1, aid2), score in pairs_this_batch[row_i].items():
            if aid1 not in fullSimMatrix: 
                fullSimMatrix[aid1] = {0: 0.0 for _ in range(0)}
            if aid2 not in fullSimMatrix[aid1]:
                fullSimMatrix[aid1][aid2] = 0.0
            fullSimMatrix[aid1][aid2] += score


# =======================================
# 1. Select top items to recommend in re-ranking
# 2. Compute Real time importance of each action (Not in use currently).
# =======================================
@nb.jit(nopython = True)
def heap_topk_return_list(item_cnt_dict, cap):
    """
    get the top cap(k) elements of the cnt dict based on value, using a min-heap structure, return a list with top "cap" elements with highest score
    """
    q = [(np.float64(0), np.int64(0)) for _ in range(0)]  ## generate empty queue to implement a heap, 
    for item_ref, sim_score in item_cnt_dict.items():   ## read in the dict in heap structure
        heapq.heappush(q, (sim_score, item_ref))   ## push the <sim_score, item_ref_id> pair into min-heap, using sim_score for order
        if len(q) > cap:
            heapq.heappop(q)
            
    res = [heapq.heappop(q)[1] for _ in range(len(q))][::-1]
    
    return res

# @nb.jit(nopython=True)
# def getRealTimeActionWeight(ts_action, ts_start, ts_end, op, seq, length, ACTION_WEIGHTS, TRAIN_START_TS, TEST_END_TS):
#     """
#     This function returns the real time importance weight of test set session actions
#     input: 
#         ts_action: ts of the action takes place
#         ts_start: start_ts of this session
#         ts_end: end_ts of this session
#         op: type of the action
#         seq: seq order this action
#         length: total # of actions of this session. 
#     """
#     action_weight = ACTION_WEIGHTS[op]
#     sequence_weight = max(2 ** (seq/length) - 1, 0.5) ## floor at 0.1  #np.power(2, np.linspace(seq_weight_const, 1, length))[::-1] - 1
#     res = action_weight * sequence_weight # * time_weight
    
#     return res