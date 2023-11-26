import pandas as pd

# CSV 파일 읽기
df1 = pd.read_csv('DinoV2_MSE.csv')
df2 = pd.read_csv('Jetson_MSE.csv')

# 두 DataFrame을 행 방향으로 합치기
combined_df = pd.concat([df1, df2])

# 결과를 새로운 CSV 파일로 저장
combined_df.to_csv('DinoV2_Jetson_MSE.csv', index=False)