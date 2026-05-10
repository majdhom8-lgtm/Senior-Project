# ═══════════════════════════════════════════════════════════════════════════════
#  tools/data_reader.py
#  الأداة 1: DataReaderTool
#  المرجع: InfiAgent-DABench — Hu et al., ICML 2024 — arXiv:2401.05507
#
#  المهمة: قراءة ملف CSV/Excel وإنتاج تقرير شامل يُغذّي الحلقة الأجنتية
# ═══════════════════════════════════════════════════════════════════════════════

import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path


SUPPORTED = {".csv", ".xlsx", ".xls", ".tsv"}


class DataReaderTool:

    def run(self, file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            return f"[خطأ] الملف غير موجود: {file_path}"
        if path.suffix.lower() not in SUPPORTED:
            return f"[خطأ] صيغة غير مدعومة: {path.suffix}. المدعومة: {SUPPORTED}"

        try:
            df = self._read(path)
        except Exception as e:
            return f"[خطأ] فشل في القراءة: {e}"

       
        cache = os.path.join(tempfile.gettempdir(), f"agt_{path.stem}.pkl")
        df.to_pickle(cache)

        return self._report(df, file_path)


    def _read(self, path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext == ".csv":
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    return pd.read_csv(path, encoding=enc)
                except UnicodeDecodeError:
                    continue
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"صيغة غير معروفة: {path.suffix}")


    def _report(self, df: pd.DataFrame, path: str) -> str:
        lines = []
        add = lines.append

        add(f"{'='*60}")
        add(f"تقرير قراءة البيانات: {path}")
        add(f"{'='*60}")
        add(f"الشكل : {df.shape[0]:,} صف  ×  {df.shape[1]} عمود")
        add(f"الحجم : {df.memory_usage(deep=True).sum()/1024:.1f} KB")

        
        add("\n[ الأعمدة وأنواعها ]")
        for col in df.columns:
            miss     = df[col].isna().sum()
            miss_pct = miss / len(df) * 100
            uniq     = df[col].nunique()
            sev      = "🔴 عالي" if miss_pct > 30 else "🟡 متوسط" if miss_pct > 5 else "🟢 منخفض"
            add(f"  {col:<25} نوع={str(df[col].dtype):<10} "
                f"مفقود={miss}({miss_pct:.1f}%){sev}  فريد={uniq}")

        
        dups = df.duplicated().sum()
        add(f"\n[ الصفوف المكررة ]: {dups} ({dups/len(df)*100:.1f}%)")

        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            add("\n[ إحصاءات الأعمدة الرقمية ]")
            add(df[num_cols].describe().round(2).to_string())

        
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            add("\n[ أعمدة نصية — أعلى القيم تكراراً ]")
            for col in cat_cols[:6]:
                top = df[col].value_counts().head(4)
                vals = "  |  ".join([f"{v}({c})" for v, c in top.items()])
                add(f"  {col}: {vals}")

        
        add("\n[ عينة (5 صفوف أولى) ]")
        add(df.head(5).to_string(index=False))

        
        add("\n[ توصيات أولية للتنظيف ]")
        recs = self._recommendations(df)
        for r in recs:
            add(f"  ⚡ {r}")

        return "\n".join(lines)

    def _recommendations(self, df: pd.DataFrame) -> list:
        recs = []
        miss = df.isnull().sum() / len(df) * 100

        hi = miss[miss > 30].index.tolist()
        if hi:
            recs.append(f"أعمدة بمفقود عالٍ جداً (>30%) — فكّر في حذفها: {hi}")

        med = miss[(miss > 0) & (miss <= 30)].index.tolist()
        if med:
            recs.append(f"أعمدة تحتاج imputation: {med}")

        if df.duplicated().sum() > 0:
            recs.append(f"يوجد {df.duplicated().sum()} صف مكرر — يُنصح بإزالتها")

        for col in df.select_dtypes(include=["object"]).columns:
            kws = ["date","time","year","month","day","تاريخ","وقت"]
            if any(k in col.lower() for k in kws):
                recs.append(f"العمود '{col}' يبدو تاريخاً — تحقق من تحويله لـ datetime")

        for col in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            out = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
            if out > 0:
                recs.append(f"العمود '{col}' يحتوي {out} شذوذ خارج 3×IQR")

        if not recs:
            recs.append("البيانات تبدو جيدة نسبياً — لا مشاكل جوهرية مكتشفة")
        return recs
