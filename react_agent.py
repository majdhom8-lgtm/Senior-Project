# ═══════════════════════════════════════════════════════════════════════════════
#  agent/react_agent.py
#  محرك الاستدلال — حلقة ReAct
#  المرجع: Yao et al., "ReAct: Synergizing Reasoning and Acting"
#           ICLR 2023 — arXiv:2210.03629
#
#  الحلقة: THOUGHT → ACTION → OBSERVATION → THOUGHT → ... → FINAL ANSWER
#  مع: Self-Debugging من DatawiseAgent (EMNLP 2024)
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json
import os
import logging
from datetime import datetime
from openai import OpenAI

from tools.data_reader     import DataReaderTool
from tools.sales_predictor import SalesPredictorTool
from tools.data_cleaner import DataCleaningTool
from tools.sandbox      import PythonSandbox
from memory            import SessionMemory, ConversationMemory
import config

# ── إعداد التسجيل (Logging) ──────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ReactAgent')


# ── الأدوات المتاحة للعميل ────────────────────────────────────────────────────
TOOLS = {
    "read_data"   : DataReaderTool(),
    "clean_data"  : DataCleaningTool(),
    "execute_code"  : PythonSandbox(),
    "predict_sales" : SalesPredictorTool(),
}

# ── وصف الأدوات يُرسَل في System Prompt ──────────────────────────────────────
TOOLS_DESC = """
الأدوات المتاحة:

1. read_data(file_path)
   المهمة : يقرأ ملف CSV أو Excel ويُنتج تقريراً شاملاً عن البيانات
   المخرج : أعمدة، أنواع، قيم مفقودة، إحصاءات، شذوذات، توصيات
   مثال   : ACTION: read_data
             INPUT: {"file_path": "data/sales.csv"}

2. clean_data(file_path, target_columns, analysis_goal)
   المهمة : يُنظّف البيانات بمنطق AutoDCWorkflow الثلاثي (Purpose-Driven)
             المرحلة 1: تحديد الأعمدة ذات الصلة
             المرحلة 2: فحص الجودة (Standardization / Error Correction / Missing Imputation)
             المرحلة 3: توليد وتنفيذ عمليات التنظيف
   مثال   : ACTION: clean_data
             INPUT: {"file_path": "data/sales.csv", "target_columns": ["Sales", "Profit"], "analysis_goal": "تحليل الأرباح"}

3. execute_code(code, dataframe_name)
   المهمة : ينفّذ كود Python في بيئة آمنة معزولة مع الوصول لـ pandas, numpy
   مثال   : ACTION: execute_code
             INPUT: {"code": "print(df.shape)", "dataframe_name": "sales"}

4. predict_sales(category, sub_category, region, ship_mode, quantity, discount)
   المهمة : يتوقع المبيعات المستقبلية باستخدام نموذج RandomForest مُدرَّب على Superstore (Kaggle)
   الفئات : Technology / Furniture / Office Supplies
   المناطق: West / East / Central / South
   الشحن  : Standard Class / First Class / Second Class / Same Day
   مثال   : ACTION: predict_sales
             INPUT: {"category": "Technology", "sub_category": "Phones", "region": "West", "ship_mode": "Standard Class", "quantity": 3, "discount": 0.1}

قواعد صيغة الاستدعاء (يجب الالتزام بها تماماً):
  ACTION: اسم_الأداة
  INPUT: {"مفتاح": "قيمة"}

عند الانتهاء اكتب:
  FINAL ANSWER:
  [تقريرك الشامل هنا]
"""

SYSTEM_PROMPT = f"""أنت محلل بيانات ذكي وكيل (Agentic AI Data Analyst).
مهمتك في هذا الفصل: قراءة البيانات الخام ومعالجتها وتنظيفها.

أسلوب عملك يتبع إطار ReAct (Reasoning + Acting) من ورقة ICLR 2023:
  THOUGHT    : فكّر بصوت عالٍ — ما تعرفه؟ ما تحتاجه؟ ما خطوتك التالية؟
  ACTION     : نفّذ الأداة المناسبة
  OBSERVATION: اقرأ النتيجة وابنِ عليها في THOUGHT التالية
  كرّر حتى تصل لـ FINAL ANSWER

{TOOLS_DESC}

قواعد مهمة:
- كل قرار تنظيف يجب أن يكون مُبرَّراً في الـ THOUGHT
- إذا فشل شيء، حلّل السبب وجرّب نهجاً مختلفاً (Self-Debugging)
- لا تُنهِ المهمة قبل الانتهاء من التنظيف
- التقرير النهائي يجب أن يشرح بالعربية كل ما تم
"""


class ReactAgent:
    """
    حلقة ReAct الأجنتية.
    THOUGHT → ACTION → OBSERVATION → ... → FINAL ANSWER
    """

    def __init__(self):
        self.client  = OpenAI(api_key=config.OPENAI_API_KEY)
        self.history = []
        self.log     = []       # سجل كامل لكل الخطوات
        self.retry   = 0        # عداد التصحيح الذاتي

    # ── نقطة الدخول الرئيسية ─────────────────────────────────────────────────
    def run(self, goal: str) -> dict:
        print(f"\n{'═'*60}")
        print(f"  الهدف: {goal}")
        print(f"{'═'*60}")

        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": goal},
        ]

        for step in range(1, config.MAX_STEPS + 1):
            print(f"\n{'─'*50}")
            print(f"  الخطوة {step}")
            print(f"{'─'*50}")

            # ── استدعاء النموذج ───────────────────────────────────────────────
            response = self.client.chat.completions.create(
                model       = config.MODEL,
                messages    = self.history,
                temperature = config.TEMPERATURE,
                max_tokens  = config.MAX_TOKENS,
            )
            agent_text = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": agent_text})

            self._print_agent(agent_text)

            # ── FINAL ANSWER ──────────────────────────────────────────────────
            if "FINAL ANSWER:" in agent_text:
                answer = agent_text.split("FINAL ANSWER:")[-1].strip()
                print(f"\n{'═'*60}")
                print("  التقرير النهائي:")
                print(f"{'═'*60}")
                print(answer)
                return {"status": "success", "answer": answer,
                        "steps": self.log, "total": step}

            # ── تحليل الـ ACTION ─────────────────────────────────────────────
            action, inp = self._parse(agent_text)

            if not action:
                # لم يستدعِ أداة — اطلب منه المتابعة
                self.history.append({
                    "role"   : "user",
                    "content": "استمر في العمل. إذا انتهيت اكتب FINAL ANSWER: ثم تقريرك."
                })
                continue

            # ── تنفيذ الأداة ─────────────────────────────────────────────────
            obs, ok = self._execute(action, inp)

            # Self-Debugging (من DatawiseAgent)
            if not ok:
                self.retry += 1
                if self.retry >= config.MAX_RETRIES:
                    obs = f"[تجاوز الحد الأقصى للمحاولات] {obs}"
                    self.retry = 0
                else:
                    obs = (f"[فشل — المحاولة {self.retry}/{config.MAX_RETRIES}]\n"
                           f"{obs}\n"
                           f"حلّل الخطأ وجرّب نهجاً مختلفاً.")
            else:
                self.retry = 0

            # تسجيل الخطوة
            self.log.append({
                "step"   : step,
                "thought": self._get_thought(agent_text),
                "action" : action,
                "input"  : inp,
                "obs"    : obs[:400],
                "ok"     : ok,
            })

            # إضافة الـ OBSERVATION للمحادثة
            self.history.append({"role": "user", "content": f"OBSERVATION:\n{obs}"})

            preview = obs[:350] + ("..." if len(obs) > 350 else "")
            print(f"\nOBSERVATION:\n{preview}\n")

        return {"status": "max_steps", "answer": "تجاوز الحد الأقصى للخطوات",
                "steps": self.log, "total": config.MAX_STEPS}

    # ── مساعدات داخلية ───────────────────────────────────────────────────────
    def _parse(self, text: str):
        """استخراج ACTION و INPUT من نص العميل."""
        act_m = re.search(r"ACTION\s*:\s*(\w+)", text)
        inp_m = re.search(r"INPUT\s*:\s*(\{.*?\})", text, re.DOTALL)

        if not act_m:
            return None, {}

        action = act_m.group(1).strip()
        inp    = {}
        if inp_m:
            raw = inp_m.group(1).replace("'", '"')
            # تنظيف أسطر متعددة
            raw = re.sub(r"\s+", " ", raw)
            try:
                inp = json.loads(raw)
            except Exception:
                pass

        return action, inp

    def _execute(self, action: str, inp: dict):
        """تنفيذ الأداة وإرجاع (نتيجة، نجاح)."""
        if action not in TOOLS:
            return (f"الأداة '{action}' غير موجودة. المتاحة: {list(TOOLS.keys())}", False)
        try:
            result = TOOLS[action].run(**inp)
            return result, True
        except Exception as e:
            return f"استثناء في '{action}': {e}", False

    def _get_thought(self, text: str) -> str:
        m = re.search(r"THOUGHT\s*:(.*?)(?:ACTION\s*:|$)", text, re.DOTALL)
        return m.group(1).strip() if m else text[:200]

    def _print_agent(self, text: str):
        """طباعة مرتبة لمخرجات العميل."""
        if "THOUGHT:" in text:
            t = self._get_thought(text)
            print(f"THOUGHT:\n  {t[:300]}")
        am = re.search(r"ACTION\s*:\s*(\w+)", text)
        if am:
            print(f"\nACTION: {am.group(1)}")
        im = re.search(r"INPUT\s*:\s*(\{.*?\})", text, re.DOTALL)
        if im:
            print(f"INPUT:  {im.group(1)[:200]}")
