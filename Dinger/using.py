import pandas as pd
import torch
from resnet import ResNet

# Test data
data = pd.read_csv("/Users/seongyeon-kim/Desktop/Dinger/data/stock_kr/321260.csv")
data = data.dropna()

# 가장 최근 64 길이 (이미지 1장)
test_data = data.iloc[-64:][["종가", "거래량"]]

# 가장 최근 64 길이 (이미지 2장)
test_data2 = data.iloc[-65:][["종가", "거래량"]]

# 가장 최근 65 길이 (이미지 3장)
test_data3 = data.iloc[-66:][["종가", "거래량"]]

# model
model = ResNet()

# model weight load
model.load_state_dict(torch.load("/Users/seongyeon-kim/Desktop/Dinger/Dinger/resnet.pth"))

import numpy as np
# 추론 이미지 1장 (0: Up, 1: Down, 2: Side)
# Input : pd.DataFrame
# Output: torch.Tensor
pred = model.predict(test_data)
print("한장을 예측해보자", np.array(pred)[0])

# 추론 이미지 2장
pred2 = model.predict(test_data2)
print("두장을 예측해보자", pred2)

# 추론 이미지 3장
pred3 = model.predict(test_data3)
print("세장을 예측해보자", pred3)