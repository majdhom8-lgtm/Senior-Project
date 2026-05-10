# ═══════════════════════════════════════════════════════════════════════════════
#  generate_sample_data.py
#  يُولّد ملف بيانات مالية تجريبي بمشاكل مقصودة لاختبار العميل
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 300

def generate():
    os.makedirs("data", exist_ok=True)

    data = {
        "Order_ID"     : [f"ORD-{i:04d}" for i in range(1, N+1)],
        "Order_Date"   : pd.date_range("2023-01-01", periods=N, freq="D").astype(str),
        "Customer"     : np.random.choice(["Alice","Bob","Charlie","Diana",None], N),
        "Category"     : np.random.choice(["Electronics","Furniture","office supplies","Technology",None], N),
        "Sub_Category" : np.random.choice(["Phones","Chairs","Paper","CHAIRS","phones"], N),
        "Sales"        : np.random.uniform(10, 5000, N).round(2),
        "Profit"       : np.random.uniform(-500, 2000, N).round(2),
        "Quantity"     : np.random.randint(1, 20, N),
        "Discount"     : np.random.choice([0.0, 0.1, 0.2, 0.3, None], N),
        "Region"       : np.random.choice(["East","West","North","south","EAST","West "], N),
    }
    df = pd.DataFrame(data)

    # ── مشاكل مقصودة ────────────────────────────────────────────────────────
    # 1. قيم مفقودة في أعمدة رئيسية
    df.loc[10:25,  "Sales"]    = None
    df.loc[40:48,  "Profit"]   = None
    df.loc[60:65,  "Discount"] = None

    # 2. صفوف مكررة (أول 8 صفوف تتكرر)
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)

    # 3. شذوذات واضحة
    df.loc[50, "Sales"]  = 9_999_999.0   # شذوذ ضخم
    df.loc[51, "Profit"] = -99_999.0     # شذوذ سلبي

    # 4. عمود بمفقود شديد
    df["Old_System_Code"] = None   # 100% مفقود

    df.to_csv("data/financial_sample.csv", index=False, encoding="utf-8-sig")

    print("✓ تم إنشاء: data/financial_sample.csv")
    print(f"  الشكل   : {df.shape[0]} صف × {df.shape[1]} عمود")
    print(f"  المشاكل : قيم مفقودة، صفوف مكررة، شذوذات، نصوص غير موحدة، عمود فارغ كلياً")
    return "data/financial_sample.csv"

if __name__ == "__main__":
    generate()
