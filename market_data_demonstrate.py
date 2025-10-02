import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

path = "Nat_Gas.csv"
df = pd.read_csv(path)

date_col = df.columns[0]
price_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df[[date_col, price_col]].dropna().rename(columns={date_col: "date", price_col: "price"})
df = df.sort_values("date").reset_index(drop=True)

df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)
df["month"] = df["date"].dt.month

month_dummies = pd.get_dummies(df["month"], prefix="m").iloc[:, 1:]
X = pd.concat([df[["t"]], month_dummies], axis=1)
y = df["price"].values

lr = LinearRegression()
lr.fit(X, y)

df["trend_seasonal"] = lr.predict(X)

resid = y - df["trend_seasonal"]
resid_model = sm.tsa.ARIMA(resid, order=(1,0,0)).fit()

phi = resid_model.params.get("ar.L1", 0.0)
const = resid_model.params.get("const", 0.0)

df["resid_fit"] = resid_model.predict(start=0, end=len(resid)-1)
df["final_fit"] = df["trend_seasonal"] + df["resid_fit"]

last_date = df["date"].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=12, freq="M")

future_t = (future_dates - df["date"].min()).days.astype(int)
future_months = future_dates.month
future_dummies = pd.get_dummies(future_months, prefix="m")
for col in month_dummies.columns:
    if col not in future_dummies.columns:
        future_dummies[col] = 0

future_X = pd.concat([pd.DataFrame({"t": future_t}), future_dummies[month_dummies.columns]], axis=1)
future_trend = lr.predict(future_X)

last_resid = resid.iloc[-1]
future_resid = []
r = last_resid
for _ in range(len(future_dates)):
    r = const + phi * r
    future_resid.append(r)

future_final = future_trend + np.array(future_resid)

future_df = pd.DataFrame({"date": future_dates, "price": np.nan, "final_fit": future_final})
combined = pd.concat([df[["date","price","final_fit"]], future_df], ignore_index=True)

def estimate_price(date_input):
    """Estimate price for any given date (string or datetime)."""
    q = pd.to_datetime(date_input)
    ref = combined.set_index("date")["final_fit"]

    if q in ref.index:
        return float(ref.loc[q])

    if q < ref.index.min() or q > ref.index.max():
        if q < ref.index.min():
            # Backward extrapolation using trend+seasonality only
            t_q = (q - df["date"].min()).days
            m_q = q.month
            md = {col:0 for col in month_dummies.columns}
            col = f"m_{m_q}"
            if col in md: md[col] = 1
            Xq = np.array([t_q] + [md[c] for c in month_dummies.columns]).reshape(1,-1)
            return float(lr.predict(Xq)[0])
        else:
            # Beyond last date: linear interpolation between computed forecast endpoints
            idx = ref.index
            idx_before = idx[idx <= q].max()
            idx_after = idx[idx >= q].min()
            v1, v2 = ref.loc[idx_before], ref.loc[idx_after]
            frac = (q - idx_before).days / (idx_after - idx_before).days
            return float(v1 + frac*(v2-v1))

    idx = ref.index
    idx_before = idx[idx <= q].max()
    idx_after = idx[idx >= q].min()
    if idx_before == idx_after:
        return float(ref.loc[idx_before])
    v1, v2 = ref.loc[idx_before], ref.loc[idx_after]
    frac = (q - idx_before).days / (idx_after - idx_before).days
    return float(v1 + frac*(v2-v1))

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["price"], "x", label="Observed")
plt.plot(combined["date"], combined["final_fit"], "o-", label="Model + Forecast")
plt.title("Natural Gas Prices (Observed & Forecast)")
plt.xlabel("Date"); plt.ylabel("Price")
plt.legend(); plt.grid(True); plt.show()

print("Estimate on 2021-07-15:", estimate_price("2021-07-15"))
print("Estimate on 2023-12-01:", estimate_price("2023-12-01"))
print("Estimate on 2024-09-30:", estimate_price("2024-09-30"))
print("Estimate on 2025-02-15:", estimate_price("2025-02-15"))
