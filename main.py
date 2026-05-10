# ═══════════════════════════════════════════════════════════════════════════════
#  main.py — نقطة الدخول الرئيسية
#  Agentic AI Data Analyst — الفصل الأول
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from agent.react_agent import ReactAgent
from generate_sample_data import generate as make_sample


def check_api_key():
    if config.OPENAI_API_KEY == "ضع_مفتاحك_هنا" or not config.OPENAI_API_KEY:
        print("═" * 55)
        print("  خطأ: لم تضع مفتاح OpenAI API")
        print("  افتح ملف config.py وضع مفتاحك في السطر الأول")
        print("═" * 55)
        sys.exit(1)


def build_goal(file_path: str, user_goal: str) -> str:
    """بناء الهدف الكامل الذي يُرسَل للعميل."""
    return f"""
ملف البيانات: {file_path}

المهمة: {user_goal}

الخطوات المقترحة:
1. استخدم read_data لقراءة الملف وفهم محتواه ومشاكله
2. في الـ THOUGHT الثانية: حلّل تقرير القراءة وحدد أولويات التنظيف
3. استخدم clean_data لتنظيف البيانات بناءً على تحليلك
4. تحقق من نتيجة التنظيف بـ execute_code إذا احتجت
5. اكتب FINAL ANSWER بتقرير عربي شامل يشرح كل ما تم
""".strip()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic AI Data Analyst — الفصل الأول"
    )
    parser.add_argument("--file",   type=str, help="مسار ملف البيانات (CSV/Excel)")
    parser.add_argument("--goal",   type=str, default="اقرأ هذه البيانات وحدد مشاكلها ونظّفها بشكل ذكي ومُبرَّر",
                        help="هدف التحليل")
    parser.add_argument("--sample", action="store_true",
                        help="استخدام بيانات تجريبية مالية")
    args = parser.parse_args()

    check_api_key()

    
    if args.sample or not args.file:
        print("جاري إنشاء بيانات تجريبية مالية...")
        file_path = make_sample()
    else:
        if not Path(args.file).exists():
            print(f"خطأ: الملف غير موجود: {args.file}")
            sys.exit(1)
        file_path = args.file


    print(f"\n{'═'*55}")
    print(f"  Agentic AI Data Analyst — الفصل الأول")
    print(f"  النموذج : {config.MODEL}")
    print(f"  الملف   : {file_path}")
    print(f"  الهدف   : {args.goal}")
    print(f"{'═'*55}")

    agent  = ReactAgent()
    goal   = build_goal(file_path, args.goal)
    result = agent.run(goal)


    print(f"\n{'═'*55}")
    print(f"  انتهى في {result['total']} خطوة | الحالة: {result['status']}")
    print(f"{'═'*55}")

    if result["status"] == "success":
        print("\n✓ تم تنفيذ المهمة بنجاح.")
        cleaned = Path(file_path)
        cleaned_path = cleaned.parent / f"{cleaned.stem}_cleaned{cleaned.suffix}"
        if cleaned_path.exists():
            print(f"✓ الملف النظيف: {cleaned_path}")
    else:
        print("\n⚠ تجاوز الحد الأقصى للخطوات.")


if __name__ == "__main__":
    main()
