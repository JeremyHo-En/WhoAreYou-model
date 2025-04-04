import pandas as pd

# 初始化結果容器
all_results = []
df = pd.read_csv('./ERC-20_reset.csv')
contracts = df['Address'].dropna().unique().tolist()

temp = 0
for i in contracts:
    try:
        # 假設 fetch_token_feature_and_risk(i, API_KEY) 會返回資料
        results = fetch_token_feature_and_risk(i, API_KEY)
        
        # 每次成功抓取資料後，即時將其加入結果列表
        all_results.append(results)

        # 即時轉換並儲存為 CSV
        temp_df = pd.DataFrame([results])  # 將當前結果包裝為 DataFrame
        temp_df.to_csv("./token_features_with_risk_0840.csv", mode='a', header=temp == 0, index=False)

    except Exception as e:
        print(f"第 {i} 批發生錯誤：{e}")

    # 進行計數
    temp += 1
    print(f"處理進度: {temp}/{len(contracts)}")

# 完成後提示
print("✅ 已完成並輸出 token_features_with_risk_0840.csv")
