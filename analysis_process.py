import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. データクレンジング (ドメイン知識に基づく外れ値除外)
# ==========================================
# データの読み込み
df = pd.read_csv('data/raw/SSDSE-A-2025.csv')

# 欠損値（震災による統計不連続地域など）の除外
# '-' や '***' などの非数値をNaNに変換し、欠損値を含む行をドロップ
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# 人口5,000人未満の自治体を除外（極小N数によるノイズ排除）
df_filtered = df[df['総人口'] >= 5000].copy()

# ==========================================
# 2. 特徴量エンジニアリング (行動遺伝学的変数の構築)
# ==========================================
# 環境指標（共有環境：C）
df_filtered['1人あたり教育費'] = df_filtered['教育費'] / df_filtered['15歳未満人口']

# 適応指標（表現型：P）
df_filtered['第3次産業比率'] = (df_filtered['第3次産業就業者数'] / df_filtered['就業者数']) * 100

# 環境選択の指標（能動的rGE）
df_filtered['転入超過率'] = ((df_filtered['転入者数'] - df_filtered['転出者数']) / df_filtered['総人口']) * 100

# 制御変数
df_filtered['可住地人口密度'] = df_filtered['総人口'] / df_filtered['可住地面積']

# ==========================================
# 3. 統計的モデリング (重回帰分析)
# ==========================================
# 独立変数 (X) と 従属変数 (Y) の定義
X = df_filtered[['第3次産業比率', '可住地人口密度', '1人あたり教育費']]
X = sm.add_constant(X) # 切片を追加
Y = df_filtered['転入超過率']

# OLS（通常最小二乗法）モデルの構築と学習
model = sm.OLS(Y, X).fit()

# 分析結果（β係数、P値など）の表示
print(model.summary())

# ==========================================
# 4. 残差分析と特異点の抽出 (南幌町モデルの発見)
# ==========================================
# 予測値と標準化残差を取得
df_filtered['予測値'] = model.fittedvalues
df_filtered['標準残差'] = model.resid_pearson

# 「能動的rGE」の証拠となる、標準残差が異常に高い自治体を抽出（例：南幌町）
outliers = df_filtered[df_filtered['標準残差'] > 3.0]
print("--- 特異的適応を示す自治体 (標準残差 > 3) ---")
print(outliers[['都道府県', '市区町村', '標準残差']])

# ==========================================
# 5. 可視化 (予測値 vs 標準残差の散布図)
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(x='予測値', y='標準残差', data=df_filtered)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted (Environmental vs Phenotypic Variance)')
plt.xlabel('Predicted Net Migration Rate')
plt.ylabel('Standardized Residuals')
plt.show()
