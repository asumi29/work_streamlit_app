import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'

st.subheader('確率分布の実験')

"""
### 正規分布
母数（パラメータ）を変化させたときのグラフの変化の確認
"""

mu = st.sidebar.slider('正規分布の期待値', min_value=-5.0, max_value=5.0, step=0.01, value=0.0)
sigma = st.sidebar.slider('正規分布の分散', min_value=0.1, max_value=20.0, step=0.1, value=1.0)

x_1 = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
z = stats.norm.pdf(x_1, loc=mu, scale=sigma)

fig_norm, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x_1, z, 'b-', linewidth=2, label='正規分布')
ax1.fill_between(x_1, z, alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('確率密度関数')
ax1.set_title(f'N({mu:.2f}, {sigma:.2f}^2)')
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig_norm, use_container_width=True)

"""
### ポアソン分布
母数（パラメータ）を変化させたときのグラフの変化の確認
"""

lam = st.sidebar.slider('ポアソン分布の期待値 λ', min_value=0, max_value=30, step=1, value=5)

x_2 = np.arange(0, 31)
r = stats.poisson.pmf(x_2, mu=lam)

fig_pois, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(x_2, r, color='#00A968', alpha=0.7, edgecolor='black', label=f'Poisson(λ={lam})')
ax2.set_xlabel('k (離散値)')
ax2.set_ylabel('確率質量関数')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
st.pyplot(fig_pois, use_container_width=True)

st.subheader('数理モデルの実験')

def SIR(t, y, beta, gamma):
    """SIRモデル微分方程式"""
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# パラメータ
T = 200
S0, I0, R0 = 999, 1, 0

r0 = st.sidebar.slider('基本再生産数 R₀', min_value=1.0, max_value=10.0, step=0.01, value=2.0)
recovery_days = st.sidebar.slider('回復期間 (日)', min_value=1, max_value=20, step=1, value=7)
gamma = 1 / recovery_days
beta = r0 * gamma / S0

st.info(f"**感染率 β**: {beta:.4f}/人/日\n**回復率 γ**: {gamma:.3f}/日")

# SIRシミュレーション
t_eval = np.linspace(0, T, 1000)
sol = solve_ivp(SIR, [0, T], [S0, I0, R0], args=(beta, gamma), t_eval=t_eval)

fig_sir, ax_sir = plt.subplots(figsize=(10, 6))
ax_sir.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='感受性者 (S)')
ax_sir.plot(sol.t, sol.y[1], 'r-', linewidth=2, label='感染者 (I)')
ax_sir.plot(sol.t, sol.y[2], 'g-', linewidth=2, label='回復者 (R)')
ax_sir.set_xlabel('日数')
ax_sir.set_ylabel('人数')
ax_sir.set_title(f'SIRモデル (R₀={r0:.2f}, 回復期間={recovery_days}日)')
ax_sir.legend()
ax_sir.grid(True, alpha=0.3)
ax_sir.set_ylim(0, 1100)

# ピーク感染者数表示
peak_i = np.max(sol.y[1])
peak_day = sol.t[np.argmax(sol.y[1])]
st.metric("最大感染者数", f"{peak_i:.0f}人", f"{peak_day:.0f}日目")

st.pyplot(fig_sir, use_container_width=True)




