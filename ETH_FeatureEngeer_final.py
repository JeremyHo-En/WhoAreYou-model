#%%
from web3 import Web3
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
from requests.exceptions import SSLError
import os

#%%
def extract_features(df, wallet_address,tcount_trans):
    features = {}
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    ETH_CONVERSION = 1e18

    # 交易基本統計
    features["tx_count"] = tcount_trans  # 總交易筆數
    features["avg_value"] = df["value"].astype(float).dropna().mean() / ETH_CONVERSION  # 平均交易金額
    features["max_value"] = df["value"].astype(float).dropna().max() / ETH_CONVERSION  # 最大交易金額
    features["median_gas_price"] = df["gasPrice"].astype(float).dropna().median() / 1e9  # Gas Price（Gwei）

    # 唯一地址數
    features["unique_receivers"] = df["to"].dropna().nunique()  # 唯一收款人數
    features["unique_senders"] = df["from"].dropna().nunique()  # 唯一發送人數

    # 取得最早與最晚交易的時間戳記
    if not df["timestamp"].dropna().empty:
        min_timestamp = df["timestamp"].dropna().min().timestamp()  # 轉換為秒
        max_timestamp = df["timestamp"].dropna().max().timestamp()
        time_diff = max_timestamp - min_timestamp
    else:
        min_timestamp, max_timestamp, time_diff = 0, 0, 0

    # 計算每日交易頻率
    features["tx_per_day"] = features["tx_count"] / (time_diff / (3600 * 24)) if time_diff > 0 else 0

    # 進出交易數
    features["in_tx_count"] = df[df["to"] == wallet_address].shape[0]
    features["out_tx_count"] = df[df["from"] == wallet_address].shape[0]
    features["in_out_ratio"] = features["in_tx_count"] / max(features["out_tx_count"], 1)

    # 自己轉給自己交易比率
    features["self_tx_ratio"] = df[df["from"] == df["to"]].shape[0] / max(features["tx_count"], 1)

    # 短時間內交易爆發
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    features["max_tx_in_1h"] = df.resample("1h", on="timestamp").size().max(skipna=True)
    features["max_tx_in_5min"] = df.resample("5min", on="timestamp").size().max(skipna=True)

    # 交易時間標準差
    features["tx_time_std"] = df["timestamp"].diff().dt.total_seconds().dropna().std()

    # 交易金額異常
    features["zero_value_tx_ratio"] = (df["value"].astype(float) == 0).sum() / max(features["tx_count"], 1)
    if not df["value"].astype(float).dropna().empty:
        high_value_threshold = df["value"].astype(float).dropna().quantile(0.99)
        features["high_value_tx_ratio"] = (df["value"].astype(float) > high_value_threshold).sum() / max(features["tx_count"], 1)
    else:
        features["high_value_tx_ratio"] = 0

    # 短時間內同一個收款地址多筆交易
    df["tx_to_same_address"] = df.groupby("to")["timestamp"].diff().dt.total_seconds()
    features["short_time_repeated_tx"] = (df["tx_to_same_address"].dropna() < 60).sum() / max(features["tx_count"], 1)

    return features

# 設置 API 的基本 URL 和 headers
BASE_URL = "https://web3.nodit.io/v1/ethereum/mainnet/blockchain/getTransactionsByAccount"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": 
}

def fetch_transactions(wallet_address, max_transactions=1000):
    """抓取最多 max_transactions 筆指定錢包地址的交易資料"""
    all_transactions = []
    page = 1
    cursor = None

    while True:
        payload = {
            "accountAddress": wallet_address,
            "withCount": True,
            "withLogs": False,
            "withDecode": False,
            "fromBlock": "latest",
            "toBlock": "earliest",
            "rpp": 1000}

        if cursor:
            payload["cursor"] = cursor

        response = requests.post(BASE_URL, json=payload, headers=HEADERS)

        if response.status_code == 200:
            data = response.json()
            transactions = data.get("items", [])
            tcount_trans = data.get("count", [])
            all_transactions.extend(transactions)

            print(f"[{wallet_address}] Page {page} - Fetched {len(transactions)} tx")
            page += 1

            if int(tcount_trans) >= max_transactions:
                print(f"[{wallet_address}] Reached {max_transactions} transactions. Stopping.")
                all_transactions = all_transactions[:max_transactions]
                break

            cursor = data.get("cursor")
            if not cursor:
                print(f"[{wallet_address}] No more transactions. Stopping.")
                break

            time.sleep(1)

        elif response.status_code == 429:
            print("Rate limit exceeded. Waiting for 10 seconds...")
            time.sleep(10)
            continue
        else:
            print(f"[{wallet_address}] Error {response.status_code}: {response.text}")
            break

    df = pd.DataFrame(all_transactions)
    return df,tcount_trans if not df.empty else None

#%%

data = pd.read_csv('/Users/jeremy/Desktop/code_reset/wallet.csv')
feature_list = []
wallets = data['Address']
total = len(wallets)
temp = 0 
for wallet in wallets:

    df,tcount_trans = fetch_transactions(wallet)
    if df is not None:
        features = extract_features(df, wallet,tcount_trans)
        feature_list.append(features)
    else:
        print(f"No data for {wallet}")
    temp += 1
    print(f'{temp}/{total}')

features_df = pd.DataFrame(feature_list)
features_df.to_csv("wallet_features_ETH.csv", index=False)
# %%
