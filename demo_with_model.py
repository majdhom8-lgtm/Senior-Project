"""
demo_with_model.py
══════════════════════════════════════════════════════════════════
عرض تجريبي يُظهر العميل يستخدم الأدوات الأربع معاً:
1. read_data      ← قراءة بيانات المبيعات
2. clean_data     ← تنظيفها
3. predict_sales  ← توقع مبيعات فئات مختلفة (النموذج المستقل)
4. execute_code   ← تحليل إضافي

يعمل بدون API key لأن المخرجات محاكاة للعرض
══════════════════════════════════════════════════════════════════
"""

import sys
sys.path.insert(0, '.')

from tools.data_reader     import DataReaderTool
from tools.data_cleaner    import DataCleaningTool
from tools.sales_predictor import SalesPredictorTool
from tools.sandbox         import PythonSandbox
from generate_sample_data  import generate


def demo_full_pipeline():
    """
    محاكاة كاملة لما سيفعله العميل لو كان لديك مفتاح API
    """
    print("\n" + "═"*65)
    print("  Agentic AI Data Analyst — عرض الأدوات الأربع")
    print("═"*65)

    # ══════════════════════════════════════════════════════════════
    # الخطوة 1: العميل يقرأ البيانات
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*65)
    print("  THOUGHT: سأبدأ بقراءة ملف المبيعات لفهم بنيته")
    print("  ACTION: read_data")
    print("  INPUT: {\"file_path\": \"data/financial_sample.csv\"}")
    print("─"*65)

    file_path = generate()   # توليد بيانات تجريبية
    reader    = DataReaderTool()
    obs1      = reader.run(file_path)
    print("\nOBSERVATION (مختصر):")
    print("\n".join(obs1.split("\n")[:20]))
    print("  ...")

    # ══════════════════════════════════════════════════════════════
    # الخطوة 2: العميل يُنظّف البيانات
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*65)
    print("  THOUGHT: وجدت مشاكل — سأُنظّف البيانات أولاً")
    print("  ACTION: clean_data")
    print("─"*65)

    cleaner = DataCleaningTool()
    obs2    = cleaner.run(
        file_path,
        target_columns=["Sales", "Profit", "Category", "Region"],
        analysis_goal="تحليل وتوقع المبيعات"
    )
    print("\nOBSERVATION (مختصر):")
    print("\n".join(obs2.split("\n")[:15]))
    print("  ...")

    # ══════════════════════════════════════════════════════════════
    # الخطوة 3: العميل يستدعي نموذج التوقع
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*65)
    print("  THOUGHT: البيانات نظيفة. الآن سأستخدم نموذج التوقع")
    print("           لمقارنة فئات مختلفة وتحديد الأفضل")
    print("  ACTION: predict_sales")
    print('  INPUT: {"category": "Technology", ...}')
    print("─"*65)

    predictor = SalesPredictorTool()

    # توقع سيناريوهات متعددة
    obs3 = predictor.run(
        category="Technology",
        sub_category="Phones",
        region="West",
        scenarios=[
            {"category":"Technology",     "sub_category":"Phones",    "region":"West",    "quantity":5,  "discount":0.10},
            {"category":"Technology",     "sub_category":"Computers", "region":"East",    "quantity":2,  "discount":0.05},
            {"category":"Furniture",      "sub_category":"Chairs",    "region":"West",    "quantity":8,  "discount":0.20},
            {"category":"Office Supplies","sub_category":"Storage",   "region":"Central", "quantity":10, "discount":0.00},
            {"category":"Technology",     "sub_category":"Phones",    "region":"South",   "quantity":3,  "discount":0.30},
        ]
    )
    print("\nOBSERVATION:")
    print(obs3)

    # ══════════════════════════════════════════════════════════════
    # الخطوة 4: العميل يُنفّذ تحليل إضافي
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─"*65)
    print("  THOUGHT: سأحسب تأثير الخصم على التوقعات برمجياً")
    print("  ACTION: execute_code")
    print("─"*65)

    sandbox = PythonSandbox()
    code = """
discounts = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
print("تأثير الخصم على مبيعات Phones في West (5 وحدات):")
print(f"{'الخصم':<12} {'التوقع المقدر'}")
print("-" * 30)
for d in discounts:
    # تقدير بسيط بناءً على نسبة الخصم
    base = 2655.89
    estimated = base * (1 - d * 0.55)
    print(f"{d*100:.0f}%{'':<10} ${estimated:,.2f}")
"""
    obs4 = sandbox.run(code)
    print("\nOBSERVATION:")
    print(obs4)

    # ══════════════════════════════════════════════════════════════
    # FINAL ANSWER
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*65)
    print("  FINAL ANSWER:")
    print("═"*65)
    print("""
بعد تحليل البيانات وتنظيفها واستخدام نموذج التوقع:

📊 نتائج تحليل المبيعات:
─────────────────────────────────────────
• أعلى توقع: Phones في West بـ $2,655.89 (بدون خصم)
• أدنى توقع: Storage في Central بـ ~$531

💡 توصيات:
─────────────────────────────────────────
• التركيز على Technology في منطقة West يُعطي أعلى عائد
• الخصم 10% مقبول — يقلل المبيعات ~$220 لكن يرفع الحجم
• تجنّب خصم > 30% لأن تأثيره سلبي على الإيراد الكلي

🤖 النموذج المستخدم:
─────────────────────────────────────────
• النوع: RandomForest (Kaggle Superstore)
• الدقة: R² = 0.89 (يفسر 89% من تباين المبيعات)
• هامش الخطأ: ±$791 في المتوسط
""")

    print("═"*65)
    print("  ✓ تم باستخدام 4 أدوات: read_data + clean_data +")
    print("    predict_sales (نموذج ML) + execute_code")
    print("═"*65)


if __name__ == "__main__":
    demo_full_pipeline()
