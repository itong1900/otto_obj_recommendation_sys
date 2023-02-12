# Reranking

<!-- # <
# slot_0: is_prev_interacted, 
# slot_1: seq_w_total, 
# slot_2: time_w_total, 
# slot_3: ops_w_total: for visited item, ops weight total in this session; for unvisited item, ops weight total of the item referencing this item. 
# slot_4: session_len, 
# slot_5: num uniuqe aids, 
# slot_6: CF_score, 
# slot_7: aid's itemTotalLike: total like score use for normalization.  
# slot_8: reference time by similar matrix(if aid visited, default 100; if aid not visited, 1-19(when all aid only interact once, could grow to very large if a lot of actions on one aid), depending how many aid reference this item)
# slot_9: max_sim_score:  (1 if it's a visited item)
# slot_10: mean_sim_score: (1 if it's a visited item)
# slot_11: num_interact, (0 for unvisited item; count of interaction for visited item)
# slot_12: time_span of the session 
# slot_13: action_recency: time to last action(end time), for unvisited items -> the time to of reference_aid to the last action
# slot_14: seq_w_max: 
# slot_15: seq_w_mean: for visited item -> seq_w_total / num_interact; for unvisited item -> seq_w_total / reference_time
## ========================= round 2 ================
# slot_16: seq_w_min: 
# slot_17: time_w_max:
# slot_18: time_w_mean: similar to slot_15
# slot_19: time_w_min
# slot_20: ops_w_max:
# slot_21: ops_w_mean:
# slot_22: ops_w_min: 
# slot_23: num_clicks: visited item, direct num; unvisited item, take the reference item's num
# slot_24: num_carts:
# slot_25: num_orders:
# slot_26: last_action_type: 0 -> clicks, 1-> carts, 2 -> orders
# slot_27: time_to_now: latest interaction time to now 
# slot_28: cf_increment_max: 
# slot_29: cf_increment_mean:
# slot_30: cf_increment_min:
##  ======================== Derivable ============================
# slot_31: seq_w max_min_gap: slot_14 - slot_16 
# slot_32: time_w_max_min_gap: slot_17 - slot_19
# slot_33: ops_w_max_min_gap: slot_20 - slot_22
# slot_34: cf_incre_max_min_gap: slot_28 - slot_30
## ====================== Round 3 features ========================
# slot_35: raw_seq_order: last action's seq_order, no depreciation
# slot_36: raw_seq_order_sum: 
# slot_37: raw_seq_order_max: 
# slot_38: raw_seq_order_mean:  min not needed, as that's exactly 35
# > -->