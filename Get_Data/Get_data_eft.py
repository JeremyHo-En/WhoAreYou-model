#%%
import json
import time
import csv
from web3 import Web3
import pandas as pd
import requests

KEY = '0fe30e95afa24bb8ae20321b88efaf2e'
KEY = '54042c49f1244b009e8138a6e5b62188'
RPC_URL = f"https://mainnet.infura.io/v3/{KEY}"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# 地址分類函數
def resolve_proxy(w3, address):
    """
    嘗試解析 EIP-1967 Proxy 的邏輯合約地址。
    """
    # EIP-1967 implementation slot (固定值)
    slot = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc
    try:
        impl_bytes = w3.eth.get_storage_at(address, slot)
        impl_address = Web3.to_checksum_address(impl_bytes[-20:].hex())
        if w3.eth.get_code(impl_address) != b'':
            return impl_address
    except:
        pass
    return address


def classify_address(w3, address):
    try:
        address = Web3.to_checksum_address(address)
        code = w3.eth.get_code(address)

        # Step 1: 確認是否為合約
        if not code or code == b'0x' or code == b'':
            return 'wallet', address

        original_address = address

        # Step 2: 嘗試解析 Proxy 並取出邏輯合約
        impl_address = resolve_proxy(w3, address)
        if impl_address != address:
            print(f"{address} 是 Proxy，邏輯合約為 {impl_address}")
        address = impl_address

        # Step 3: ERC-20 判斷
        erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint256"}],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "payable": False,
                "stateMutability": "view",
                "type": "function"
            },
        ]

        contract = w3.eth.contract(address=address, abi=erc20_abi)
        try:
            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()
            return 'ERC-20', address
        except:
            pass

        # Step 4: NFT 判斷（ERC-721 / ERC-1155）
        ERC_STANDARD_IDS = {
            "ERC-721": "0x80ac58cd",
            "ERC-1155": "0xd9b67a26",
        }

        supports_interface_abi = [{
            "constant": True,
            "inputs": [{"name": "interfaceId", "type": "bytes4"}],
            "name": "supportsInterface",
            "outputs": [{"name": "", "type": "bool"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function"
        }]

        contract = w3.eth.contract(address=address, abi=supports_interface_abi)

        for standard, interface_id in ERC_STANDARD_IDS.items():
            try:
                if contract.functions.supportsInterface(interface_id).call():
                    return 'NFT', address
            except:
                pass

        # Step 5: fallback
        return 'Other', address

    except Exception as e:
        print(f"錯誤: {e}")
        return 'Error', address

def fetch_addresses(target_count=100):
    addresses = set()  # 使用集合來確保不會有重複的地址
    latest_block = w3.eth.block_number  # 當前最新區塊
    i = 0  # 計數器，用來向後遍歷區塊
    while len(addresses) < target_count and i < 100:  # 設置最大抓取次數為 100
        try:
            # 嘗試獲取區塊資料
            block = w3.eth.get_block(latest_block - i, full_transactions=True)

            # 解析區塊中的交易
            for tx in block.transactions:
                if tx["to"]:  # 如果交易有目標地址
                    addresses.add(tx["to"])
                if tx["from"]:  # 如果交易有來源地址
                    addresses.add(tx["from"])

                # 如果抓取到足夠的地址，則結束
                if len(addresses) >= target_count:
                    print(f"已抓取到 {len(addresses)} 個地址，達到目標數量")
                    return set(addresses)

            i += 1  # 移動到下一個區塊
            print(f"抓取區塊 {latest_block - i}, 當前已抓取 {len(addresses)} 個地址")
        
        except BlockNotFound:
            print(f"區塊 {latest_block - i} 未找到，稍作休息")
            time.sleep(2)  # 如果區塊未找到，休息 2 秒，防止頻繁重試
            continue
        except Exception as e:
            print(f"錯誤: {e}")
            # 判斷是否是 API 流量限制的錯誤 (例如 429 HTTP 錯誤)
            print("遇到 API 流量限制，等待一會兒...")
            time.sleep(5)  # 當出現流量限制，延長休息時間，防止過快請求

        # 根據抓取的數量進行延遲
        if len(addresses) < target_count:
            time.sleep(0.2)  # 每次抓取後稍作休息，避免過度請求

    print(f"最終抓取到的地址數量: {len(addresses)}")
    return set(addresses)


def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=["Type", "Address"])
    df.to_csv(filename, index=False)
    print(f"已儲存 {filename}")

def opensea_addresses(target_count=100):
    # 設定 API 相關參數
    url = "https://api.opensea.io/api/v2/orders/ethereum/seaport/listings"
    headers = {
        "accept": "application/json",
        "x-api-key": "d24e3c7b0cad4a388bdc1e7f343b3324"  # ⚠️ 替換成你的 API Key
    }

    # 存放唯一的 NFT 合約地址
    unique_contracts = set()
    cursor = None  # 分頁用
    target_count = target_count  # 目標收集 5000 個 NFT 合約地址
    request_count = 0  # 記錄 API 請求次數

    while len(unique_contracts) < target_count:
        # 設定查詢參數
        params = {
            "limit": 50,  # 每次最多取得 50 筆掛單
            "order_direction": "desc",
        }
        if cursor:
            params["cursor"] = cursor  # 使用分頁機制

        # 發送 API 請求
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"⚠️ API 請求錯誤: {response.status_code}, {response.text}")
            break

        data = response.json()

        # 解析 API 回應，提取 NFT 合約地址
        for order in data.get("orders", []):
            contract_address = order.get("protocol_data", {}).get("parameters", {}).get("offer", [{}])[0].get("token")

            if contract_address:
                unique_contracts.add(contract_address)

        # 更新 cursor 進行下一頁查詢
        cursor = data.get("next")

        # 記錄 API 請求次數，避免速率限制
        request_count += 1
        print(f"🔹 已收集 NFT 合約: {len(unique_contracts)} 個（API 請求次數: {request_count}）")

        # 如果沒有更多資料，停止迴圈
        if not cursor:
            print("✅ API 沒有更多掛單數據了，停止請求")
            break

        # 避免 OpenSea API 速率限制，每次請求間隔 1 秒
        time.sleep(1)

    # 最終輸出結果
    print(f"🎯 完成收集，共獲取 {len(unique_contracts)} 個 NFT 合約地址")
    return set(unique_contracts)

# 主執行函數
#%%
target_count = 10000  # 設定要抓取的地址數量
print(f"獲取最近 {target_count} 個交易的地址...")
addresses_w3 = fetch_addresses(target_count)
addresses_opensea = opensea_addresses(target_count)
addresses = set.union(addresses_w3,addresses_opensea)

classified_data = {
    "wallet": [],
    "ERC-20": [],
    "NFT": [],  # NFT 類別（包括 ERC-721 和 ERC-1155）
    "Other": []
}

progress_nb = 0
for address in addresses:
    category, addr = classify_address(w3, address)
    classified_data[category].append([category, addr])
    time.sleep(0.2)  # 避免過快請求 RPC
    progress_nb += 1
    print(f'progress : {progress_nb} / {len(addresses)}')


# 儲存分類結果
for category, data in classified_data.items():
    if data:
        save_to_csv(data, f"{category}.csv")

# %%
for category, addresses in classified_data.items():
    print(f"{category}: {len(addresses)}")
# %%

# 讀取 CSV 文件
def read_addresses_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        addresses = df['Address'].tolist()  # 假設 CSV 中有一個 'address' 欄位
        return addresses
    except Exception as e:
        print(f"讀取 CSV 文件時發生錯誤: {e}")
        return []

# 儲存分類結果
def save_to_csv(data, file_path):
    df = pd.DataFrame(data, columns=["Category", "Address"])
    df.to_csv(file_path, index=False)
    print(f"資料已儲存到 {file_path}")

# 更新分類邏輯
classified_data = {
    "wallet": [],
    "ERC-20": [],
    "NFT": [],  # NFT 類別（包括 ERC-721 和 ERC-1155）
    "Other": []
}

# 讀取 NFT 和 Other 地址
NFT_df = pd.read_csv('/Users/jeremy/Desktop/code_reset/NFT.csv')
ERC_20_df = pd.read_csv('/Users/jeremy/Desktop/code_reset/ERC-20.csv')
other_df = pd.read_csv('/Users/jeremy/Desktop/code_reset/Other.csv')

combined_df = pd.concat([NFT_df,ERC_20_df, other_df], ignore_index=True) # 保留原樣
#%%
# 分類地址
#vprogress_nb = 0
addresses = combined_df['Address'][2001:]
for address in addresses:
    category, addr = classify_address(w3, address)  # 使用你原來的 classify_address 函數
    classified_data[category].append([category, addr])
    time.sleep(0.2)  # 避免過快請求 RPC
    progress_nb += 1
    print(f'進度 : {progress_nb} / {len(addresses)}')
#%%
# 儲存分類結果
for category, data in classified_data.items():
    if data:
        save_to_csv(data, f"/Users/jeremy/Desktop/code_reset/{category}_reset.csv")
#%%
# 顯示每個類別的地址數量
for category, addresses in classified_data.items():
    print(f"{category}: {len(addresses)}")

# %%
