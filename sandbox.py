# ═══════════════════════════════════════════════════════════════════════════════
#  tools/sandbox.py
#  الأداة 3: PythonSandbox
#  المرجع: InfiAgent-DABench (ICML 2024) + DatawiseAgent (EMNLP 2024)
#
#  بيئة تنفيذ Python معزولة وآمنة
# ═══════════════════════════════════════════════════════════════════════════════

import io
import os
import sys
import traceback
import tempfile
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np

# ─── قائمة الكلمات المحظورة (أمان) ──────────────────────────────────────────
FORBIDDEN = [
    "__import__", "os.system", "os.remove", "os.unlink", "shutil",
    "subprocess", "socket", "urllib", "requests", "open(",
    "eval(", "exec(", "compile(",
]


SAFE_GLOBALS = {
    "pd"        : pd,
    "pandas"    : pd,
    "np"        : np,
    "numpy"     : np,
    "math"      : __import__("math"),
    "statistics": __import__("statistics"),
    "json"      : __import__("json"),
    "re"        : __import__("re"),
    "datetime"  : __import__("datetime"),
    "print"     : print,
    "range"     : range,
    "len"       : len,
    "list"      : list,
    "dict"      : dict,
    "str"       : str,
    "int"       : int,
    "float"     : float,
    "round"     : round,
    "sum"       : sum,
    "min"       : min,
    "max"       : max,
    "abs"       : abs,
    "sorted"    : sorted,
    "enumerate" : enumerate,
    "zip"       : zip,
    "map"       : map,
    "filter"    : filter,
}


class PythonSandbox:

    def run(self, code: str, dataframe_name: str = None) -> str:
        """
        تنفيذ كود Python في بيئة آمنة.

        المعاملات:
            code           : كود Python للتنفيذ
            dataframe_name : اسم الملف (بدون امتداد) لتحميل DataFrame تلقائياً
        """
        
        for kw in FORBIDDEN:
            if kw in code:
                return f"[خطأ أمني] الكود يحتوي على '{kw}' وهو غير مسموح."

        
        env = dict(SAFE_GLOBALS)

        if dataframe_name:
            df = self._load_df(dataframe_name)
            if df is not None:
                env["df"] = df
                env[dataframe_name] = df
            else:
                return f"[تحذير] لم يُعثر على DataFrame باسم '{dataframe_name}'. تابع بدونه."

        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            local_vars = {}
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<sandbox>", "exec"), env, local_vars)

            out   = stdout_buf.getvalue().strip()
            warns = stderr_buf.getvalue().strip()
            parts = []

            if out:
                parts.append(f"المخرجات:\n{out}")
            if warns:
                parts.append(f"تحذيرات:\n{warns}")

            
            useful_vars = {
                k: v for k, v in local_vars.items()
                if not k.startswith("_") and k not in env
            }
            for name, val in list(useful_vars.items())[:5]:
                if isinstance(val, pd.DataFrame):
                    parts.append(f"\n{name} (DataFrame {val.shape}):\n{val.head(5).to_string()}")
                elif isinstance(val, (pd.Series, np.ndarray)):
                    parts.append(f"\n{name}:\n{val}")
                elif isinstance(val, (int, float, str, list, dict, bool)):
                    parts.append(f"\n{name} = {val}")

            if not parts:
                parts.append("تم تنفيذ الكود بنجاح (لا مخرجات مطبوعة).")

            return "\n".join(parts)

        except Exception:
            tb    = traceback.format_exc()
            lines = tb.strip().splitlines()

            short = "\n".join(lines[-6:])
            return f"[خطأ في التنفيذ]\n{short}"

    def _load_df(self, name: str):
        cache = os.path.join(tempfile.gettempdir(), f"agt_{name}.pkl")
        if os.path.exists(cache):
            try:
                return pd.read_pickle(cache)
            except Exception:
                pass
        return None
