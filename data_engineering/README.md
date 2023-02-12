# Data Preprocessing with PySpark.

![image](https://user-images.githubusercontent.com/71299664/218288114-cf2c96d9-3086-458a-b56d-06fb4326b307.png)

This pipeline does the following.

### Data type transformations
1. Transform action_type to more convenient data types (clicks -> 0, carts -> 1, orders -> 2)
2. Transform timestamp in nanoseconds to seconds. This reduces a lot of memory overflows as nanoseconds is 1e9 larger and doesn't provide useful information than ts in seconds.

### Save the raw data in more readily-use format.
1. Save the meta infomation as .csv file, which contains `session_id, number_total_action, session_start_ts, session_end_ts`. 
2. Save the action info as three long int list, `ts`(flatten timestamps of all actions in each session), `ops`(flatten action_types of all actions in each session), `aids`(flatten aids of all actions in each session).

### Example usage
Query the 3rd action of session 12345.  
1. From meta_info df mentioned above, we can easily get the start idx of each session by cumsum `number_total_action`,
2. Let's session 12345 starts with idx 240001, then its 3rd action info will be stored at the idx 24003 (23001 + 3 - 1) in `ts`, `ops`, `aids`.
3. `aids[24003]`, `ts[24003]`, `ops[24003]` will tell the item_id, timestamp and interaction type of 3rd action of session 12345. 

See the diagram on top for a more staright-forward representations


