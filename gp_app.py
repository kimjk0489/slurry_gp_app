import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

st.set_page_config(page_title="Carbon Black Optimization", layout="wide")
st.title("🔬 Carbon Black 조성 최적화 (GP + Bayesian Optimization)")

# 1. 데이터 불러오기
df = pd.read_csv("C:/Dev/PythonProject/slurry_data.csv")  # 경로 확인

# 2. 입력(X), 출력(Y) 분리
x_cols = ["carbon_black_g"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 3. MinMax 정규화
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

# 4. 텐서 변환
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# 5. GP 모델 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 6. 획득함수 정의
best_y = train_y.max()
acq_fn = LogExpectedImprovement(model=model, best_f=best_y, maximize=True)
bounds = torch.stack([torch.zeros(1, dtype=torch.double), torch.ones(1, dtype=torch.double)])

# 7. 최적 조성 탐색
candidate_scaled, _ = optimize_acqf(acq_function=acq_fn, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
candidate_np = candidate_scaled.detach().numpy()
candidate_original = x_scaler.inverse_transform(candidate_np)

# 8. 추천 결과 출력
st.subheader("📌 추천된 carbon_black 조성 (단위: g)")
st.success(f"Carbon Black: {candidate_original[0][0]:.3f} g")

# 9. 시각화용 예측
x_vals = np.linspace(0, 1, 100)
X_test = torch.tensor(x_vals.reshape(-1, 1), dtype=torch.double)
model.eval()
with torch.no_grad():
    pred = model.posterior(X_test)
    mean = pred.mean.numpy().flatten()
    std = pred.variance.sqrt().numpy().flatten()

x_vals_g = x_scaler.inverse_transform(x_vals.reshape(-1, 1)).flatten()
train_x_g = x_scaler.inverse_transform(train_x.numpy()).flatten()
train_y_np = train_y.numpy().flatten()

# 10. 그래프 출력
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_vals_g, mean, color="blue", label="Predicted Mean")
ax.fill_between(x_vals_g, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% Confidence Interval")
ax.scatter(train_x_g, train_y_np, color="red", label="Observed Data")
ax.set_title("GP Prediction vs Carbon Black (unit: g)")
ax.set_xlabel("Carbon Black (g)")
ax.set_ylabel("Yield Stress")
ax.grid(True)
ax.legend()
st.pyplot(fig)
