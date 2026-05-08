"""
tools/sales_predictor.py
══════════════════════════════════════════════════════════════════
الأداة الرابعة: SalesPredictorTool
تستدعي النموذج المستقل المُدرَّب على Superstore

المهمة: توقع المبيعات لأي مجموعة من المدخلات
        أو تحليل سيناريوهات متعددة دفعةً واحدة

الاستدعاء من العميل:
    ACTION: predict_sales
    INPUT: {
        "category": "Technology",
        "sub_category": "Phones",
        "region": "West",
        "ship_mode": "Standard Class",
        "quantity": 5,
        "discount": 0.10
    }
══════════════════════════════════════════════════════════════════
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ── مسارات النموذج ──────────────────────────────────────────────────────────
MODELS_DIR    = Path("models")
MODEL_PATH    = MODELS_DIR / "sales_predictor.pkl"
ENCODER_PATH  = MODELS_DIR / "label_encoders.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# أعمدة الميزات (يجب أن تطابق ترتيب التدريب)
CATEGORICAL_COLS = ["Category", "Sub_Category", "Region", "Ship_Mode"]
NUMERICAL_COLS   = ["Quantity", "Discount"]


class SalesPredictorTool:
    """
    أداة توقع المبيعات — تُغلِّف النموذج المُدرَّب

    السمات:
        model    : النموذج المحمَّل (RandomForest/XGBoost)
        encoders : قاموس LabelEncoders للتحويل
        metadata : معلومات النموذج والأداء
    """

    def __init__(self):
        self.model    = None
        self.encoders = None
        self.metadata = None
        self._loaded  = False

    def _load_model(self):
        """تحميل النموذج عند أول استدعاء (Lazy Loading)"""
        if self._loaded:
            return True, None

        if not MODEL_PATH.exists():
            return False, (
                f"[خطأ] النموذج غير موجود في {MODEL_PATH}\n"
                f"شغّل أولاً: python train_model.py"
            )

        try:
            self.model    = joblib.load(MODEL_PATH)
            self.encoders = joblib.load(ENCODER_PATH)
            with open(METADATA_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)
            self._loaded = True
            return True, None
        except Exception as e:
            return False, f"[خطأ] فشل تحميل النموذج: {e}"

    # ══════════════════════════════════════════════════════════════════════════
    # الدالة الرئيسية — نقطة الدخول من العميل
    # ══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        category    : str,
        sub_category: str,
        region      : str,
        ship_mode   : str   = "Standard Class",
        quantity    : int   = 1,
        discount    : float = 0.0,
        scenarios   : list  = None,
    ) -> str:
        """
        توقع المبيعات لسيناريو واحد أو عدة سيناريوهات.

        المعاملات الأساسية (سيناريو واحد):
            category     : فئة المنتج  (Technology / Furniture / Office Supplies)
            sub_category : الفئة الفرعية (Phones, Chairs, Paper, ...)
            region       : المنطقة      (West / East / Central / South)
            ship_mode    : طريقة الشحن  (Standard Class / First Class / ...)
            quantity     : الكمية       (عدد صحيح > 0)
            discount     : الخصم        (0.0 إلى 0.5)

        المعاملات المتقدمة:
            scenarios    : قائمة من القواميس — لتوقع عدة سيناريوهات دفعةً
                           مثال: [{"category": "Technology", ...}, {...}]

        المخرجات:
            str : تقرير نصي بالتوقعات
        """

        # تحميل النموذج
        ok, err = self._load_model()
        if not ok:
            return err

        # وضع السيناريوهات
        if scenarios:
            return self._predict_scenarios(scenarios)
        else:
            return self._predict_single(
                category, sub_category, region,
                ship_mode, quantity, discount
            )

    # ══════════════════════════════════════════════════════════════════════════
    # توقع سيناريو واحد
    # ══════════════════════════════════════════════════════════════════════════

    def _predict_single(self, category, sub_category, region,
                        ship_mode, quantity, discount):
        """توقع المبيعات لسيناريو واحد وشرح القرار"""

        # بناء البيانات
        row = {
            "Category"    : str(category),
            "Sub_Category": str(sub_category),
            "Region"      : str(region),
            "Ship_Mode"   : str(ship_mode),
            "Quantity"    : int(quantity),
            "Discount"    : float(discount),
        }

        # تحويل إلى مصفوفة رقمية
        X, valid, warnings_list = self._encode_row(row)
        if not valid:
            return warnings_list

        # التوقع
        prediction = float(self.model.predict(X)[0])
        prediction = max(prediction, 0)  # لا مبيعات سالبة

        # تفسير التوقع
        level = self._sales_level(prediction)

        # بناء التقرير
        report = []
        report.append(f"{'='*55}")
        report.append(f"  تقرير توقع المبيعات")
        report.append(f"{'='*55}")
        report.append(f"\n  المدخلات:")
        report.append(f"  ├─ الفئة       : {category} / {sub_category}")
        report.append(f"  ├─ المنطقة     : {region}")
        report.append(f"  ├─ طريقة الشحن : {ship_mode}")
        report.append(f"  ├─ الكمية      : {quantity} وحدة")
        report.append(f"  └─ الخصم       : {discount*100:.0f}%")
        report.append(f"\n  التوقع:")
        report.append(f"  ├─ المبيعات المتوقعة : ${prediction:,.2f}")
        report.append(f"  ├─ المستوى           : {level}")
        report.append(f"  └─ المبيعات لكل وحدة : ${prediction/quantity:,.2f}")

        if warnings_list:
            report.append(f"\n  ⚠ تحذيرات:")
            for w in warnings_list:
                report.append(f"  └─ {w}")

        report.append(f"\n  معلومات النموذج:")
        report.append(f"  ├─ النوع : {self.metadata['model_name']}")
        report.append(f"  ├─ R²    : {self.metadata['best_model']['r2']}")
        report.append(f"  └─ MAE   : ±${self.metadata['best_model']['mae']:,.2f}")
        report.append(f"{'='*55}")

        # تحليل تأثير الخصم
        if discount > 0:
            pred_no_disc = self._predict_raw({**row, "Discount": 0.0})
            impact = prediction - pred_no_disc
            report.append(f"\n  تأثير الخصم: {impact:+,.2f}$ مقارنةً بدون خصم")

        return "\n".join(report)

    # ══════════════════════════════════════════════════════════════════════════
    # توقع عدة سيناريوهات
    # ══════════════════════════════════════════════════════════════════════════

    def _predict_scenarios(self, scenarios: list) -> str:
        """مقارنة سيناريوهات متعددة"""

        results = []
        for i, s in enumerate(scenarios, 1):
            row = {
                "Category"    : str(s.get("category", "Technology")),
                "Sub_Category": str(s.get("sub_category", "Phones")),
                "Region"      : str(s.get("region", "West")),
                "Ship_Mode"   : str(s.get("ship_mode", "Standard Class")),
                "Quantity"    : int(s.get("quantity", 1)),
                "Discount"    : float(s.get("discount", 0.0)),
            }
            X, valid, _ = self._encode_row(row)
            if valid:
                pred = max(float(self.model.predict(X)[0]), 0)
            else:
                pred = 0.0
            results.append((i, row, pred))

        # ترتيب تنازلي
        results.sort(key=lambda x: x[2], reverse=True)

        report = []
        report.append(f"{'='*60}")
        report.append(f"  مقارنة {len(scenarios)} سيناريو")
        report.append(f"{'='*60}")
        report.append(f"\n  {'#':<4} {'الفئة':<20} {'المنطقة':<10} "
                      f"{'الكمية':<8} {'الخصم':<8} {'التوقع':>10}")
        report.append(f"  {'-'*58}")

        for rank, (i, row, pred) in enumerate(results, 1):
            marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            report.append(
                f"  {marker} {row['Sub_Category']:<20} "
                f"{row['Region']:<10} "
                f"{row['Quantity']:<8} "
                f"{row['Discount']*100:.0f}%{'':<5} "
                f"${pred:>10,.2f}"
            )

        best = results[0]
        report.append(f"\n  ✓ الأفضل: {best[1]['Sub_Category']} في {best[1]['Region']}")
        report.append(f"  ✓ التوقع: ${best[2]:,.2f}")
        report.append(f"{'='*60}")

        return "\n".join(report)

    # ══════════════════════════════════════════════════════════════════════════
    # دوال مساعدة
    # ══════════════════════════════════════════════════════════════════════════

    def _encode_row(self, row: dict):
        """تحويل صف واحد إلى مصفوفة رقمية"""
        warnings_list = []
        encoded = {}

        # الأعمدة الفئوية
        for col in CATEGORICAL_COLS:
            le = self.encoders[col]
            val = row[col]
            if val not in le.classes_:
                warnings_list.append(
                    f"'{val}' غير موجودة في {col}. "
                    f"الخيارات: {list(le.classes_)}"
                )
                val = le.classes_[0]
            encoded[col] = le.transform([val])[0]

        # التحقق من القيم الرقمية
        qty = row["Quantity"]
        if not (1 <= qty <= 100):
            warnings_list.append(f"الكمية {qty} خارج النطاق المعقول (1-100)")
            qty = max(1, min(qty, 100))

        disc = row["Discount"]
        if not (0.0 <= disc <= 0.8):
            warnings_list.append(f"الخصم {disc} خارج النطاق (0.0-0.8)")
            disc = max(0.0, min(disc, 0.8))

        encoded["Quantity"] = qty
        encoded["Discount"] = disc

        # بناء المصفوفة
        X = np.array([[
            encoded["Category"],
            encoded["Sub_Category"],
            encoded["Region"],
            encoded["Ship_Mode"],
            encoded["Quantity"],
            encoded["Discount"],
        ]])

        return X, True, warnings_list

    def _predict_raw(self, row: dict) -> float:
        """توقع بسيط بدون تقرير"""
        X, valid, _ = self._encode_row(row)
        if not valid:
            return 0.0
        return max(float(self.model.predict(X)[0]), 0)

    def _sales_level(self, sales: float) -> str:
        """تصنيف مستوى المبيعات"""
        if sales < 100:
            return "🔴 منخفض جداً"
        elif sales < 500:
            return "🟡 منخفض"
        elif sales < 2000:
            return "🟢 متوسط"
        elif sales < 5000:
            return "🔵 مرتفع"
        else:
            return "⭐ مرتفع جداً"
