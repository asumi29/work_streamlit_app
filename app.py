import streamlit as st
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'
import numpy as np

st.subheader('確率分布の実験')

"""
### 正規分布

母数（パラメータ）を変化させたときのグラフの変化の確認
"""

# 期待値と分散を指定
mu = st.sidebar.slider('正規分布の期待値', min_value=-5.0, max_value=5.0, step=0.01, value=0.0)
sigma = st.sidebar.slider('正規分布の分散', min_value=0.1, max_value=20.0, step=0.1, value=1.0)

# 標準正規分布の描画
x_1 = np.linspace(-10, 10, 100)
z = stats.norm.pdf(x_1, loc=mu, scale=sigma)

fig_norm, ax1 = plt.subplots()
ax1.plot(x_1, z, label='std_norm')
ax1.set_xlabel('x')
ax1.set_ylabel('確率密度')
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig_norm, use_container_width=True)

"""
### ポアソン分布

母数（パラメータ）を変化させたときのグラフの変化の確認
"""

# 期待値と分散を指定
lam = st.sidebar.slider('ポアソン分布の期待値', min_value=0, max_value=30, step=1, value=5)

# ポアソン分布の描画
x_2 = np.arange(0, 31)
r = stats.poisson.pmf(x_2, mu=lam)

fig_pois, ax2 = plt.subplots()
ax2.bar(x_2, height=r, color='#00A968', alpha=0.7, label='poisson')
ax2.set_xlabel('x')
ax2.set_ylabel('確率質量')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
st.pyplot(fig_pois, use_container_width=True)

st.subheader('数理モデルの実験')

# SIRモデルの関数
def SIR(t, y, beta, gamma):
    dSdt = -beta * y[0] * y[1]
    dIdt = beta * y[0] * y[1] - gamma * y[1]
    dRdt = gamma * y[1]
    return [dSdt, dIdt, dRdt]

# 初期値の設定
T = 200  # モデルの実行期間
S0 = 999  # 感受性者数_初期値
I0 = 1   # 感染者数_初期値
R0 = 0   # 免疫保持者数_初期値

r0 = st.sidebar.slider('基本再生産数', min_value=1.0, max_value=10.0, step=0.01, value=2.0)
recovery_days = st.sidebar.slider('回復率（回復までの日数）', min_value=1, max_value=20, step=1, value=7)
gamma = 1 / recovery_days
beta = r0 * gamma / S0  # 感染率

# 分析
t_eval = np.linspace(0, T, 1000)
sol = solve_ivp(
    fun=SIR,
    t_span=[0, T],
    y0=[S0, I0, R0],
    args=(beta, gamma),
    t_eval=t_eval,
    dense_output=True,
)

# 分析結果の可視化
fig_sir, ax_sir = plt.subplots(figsize=(8, 5))
ax_sir.plot(sol.t, sol.y[0], label='Susceptible (感受性者)', linewidth=2)
ax_sir.plot(sol.t, sol.y[1], label='Infectious (感染者)', linewidth=2)
ax_sir.plot(sol.t, sol.y[2], label='Removed (回復者)', linewidth=2)
ax_sir.set_xlabel('日数 (day)')
ax_sir.set_ylabel('人数 (population)')
ax_sir.legend()
ax_sir.grid(True, alpha=0.3)
ax_sir.set_title(f'SIRモデル (R0={r0:.2f}, 回復期間={recovery_days}日)')
st.pyplot(fig_sir, use_container_width=True)

# パラメータ表示
col1, col2 = st.columns(2)
with col1:
    st.metric("基本再生産数 (R0)", f"{r0:.2f}")
with col2:
    st.metric("回復率 (γ)", f"{gamma:.3f} (1/日)")
