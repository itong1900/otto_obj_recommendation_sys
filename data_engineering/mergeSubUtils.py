import pandas as pd
import Path


def main(clicks_result_path, carts_result_path, orders_result_path, output_path):
    clicks_sub = pd.read_csv(str(clicks_result_path))
    cart_sub = pd.read_csv(str(carts_result_path))
    order_sub = pd.read_csv(str(orders_result_path))

    final_sub = pd.concat([clicks_sub, cart_sub, order_sub]).reset_index(drop=True)
    final_sub.to_csv(str(output_path), index = False)

if __name__ == '__main__':
    startTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--clicks-result', type=Path, required=True)
    parser.add_argument('--carts-result', type=Path, required=True)
    parser.add_argument('--orders-result', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    args = parser.parse_args()
    main(args.clicks_result, args.carts_result, args.orders_result, args.output_path)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))  
