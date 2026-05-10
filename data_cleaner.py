# ═══════════════════════════════════════════════════════════════════════════════
#  tools/data_cleaner.py
#  الأداة 2: DataCleaningTool
#  المرجع الأساسي : AutoDCWorkflow — Li et al., EMNLP 2025 — arXiv:2412.06724
#  المرجع النظري  : Can LLMs Clean? Survey — Zhou et al., 2026 — arXiv:2601.17058
#
#  المنطق الثلاثي (AutoDCWorkflow):
#    1. Select Target Columns  — تحديد الأعمدة ذات الصلة بالهدف
#    2. Inspect Column Quality — فحص الجودة وتصنيف المشاكل (الثلاثي من Survey)
#    3. Generate & Apply Ops   — توليد وتنفيذ عمليات التنظيف
#
#  التصنيف الثلاثي لمهام التنظيف (Can LLMs Clean Survey):
#    • Standardization    — توحيد التنسيقات
#    • Error Correction   — تصحيح الأخطاء والشذوذات
#    • Missing Imputation — معالجة القيم المفقودة
# ═══════════════════════════════════════════════════════════════════════════════

import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path


class DataCleaningTool:

    def run(
        self,
        file_path: str,
        target_columns: list = None,
        analysis_goal: str = "تحليل عام",
    ) -> str:
        """
        تنظيف البيانات بأسلوب Purpose-Driven (موجّه بهدف التحليل).

        المعاملات:
            file_path      : مسار الملف
            target_columns : الأعمدة المطلوب تنظيفها (None = كل الأعمدة)
            analysis_goal  : وصف هدف التحليل لتوجيه القرارات
        """
        
        df, err = self._load(file_path)
        if err:
            return err

        orig_shape = df.shape
        log = []
        log.append(f"{'='*60}")
        log.append(f"تقرير تنظيف البيانات")
        log.append(f"{'='*60}")
        log.append(f"الهدف    : {analysis_goal}")
        log.append(f"الشكل قبل: {orig_shape[0]:,} صف × {orig_shape[1]} عمود")

       
        if target_columns:
            valid   = [c for c in target_columns if c in df.columns]
            invalid = [c for c in target_columns if c not in df.columns]
            if invalid:
                log.append(f"[تحذير] أعمدة غير موجودة تم تجاهلها: {invalid}")
            scope = valid or list(df.columns)
        else:
            scope = list(df.columns)

        log.append(f"نطاق التنظيف: {scope}\n")


        issues = self._inspect(df, scope)

        log.append("[ فحص الجودة ]")
        if issues:
            for iss in issues:
                log.append(f"  • {iss['col']} → {iss['type']}: {iss['reason']}")
        else:
            log.append("  ✓ لم تُكتشف مشاكل")


        log.append("\n[ عمليات التنظيف المُنفَّذة ]")
        df_clean = df.copy()

        for iss in issues:
            result = self._apply(df_clean, iss)
            df_clean = result["df"]
            status  = "✓" if result["ok"] else "✗"
            log.append(f"  {status} {iss['op']} على '{iss['col']}': {result['msg']}")

        
        miss_before = df.isnull().sum().sum()
        miss_after  = df_clean.isnull().sum().sum()
        rows_removed = orig_shape[0] - df_clean.shape[0]

        log.append(f"\n[ ملخص التغييرات ]")
        log.append(f"  الشكل قبل : {orig_shape[0]:,} صف × {orig_shape[1]} عمود")
        log.append(f"  الشكل بعد : {df_clean.shape[0]:,} صف × {df_clean.shape[1]} عمود")
        if rows_removed > 0:
            log.append(f"  صفوف حُذفت: {rows_removed:,} ({rows_removed/orig_shape[0]*100:.1f}%)")
        log.append(f"  قيم مفقودة: {miss_before:,} → {miss_after:,}")

       
        out_path = self._save(df_clean, file_path)
        log.append(f"\n  ✓ حُفظت البيانات النظيفة في: {out_path}")

        return "\n".join(log)


    def _load(self, file_path: str):
        path  = Path(file_path)
        cache = os.path.join(tempfile.gettempdir(), f"agt_{path.stem}.pkl")

      
        if os.path.exists(cache):
            try:
                return pd.read_pickle(cache), None
            except Exception:
                pass

       
        try:
            ext = path.suffix.lower()
            if ext == ".csv":
                for enc in ["utf-8","utf-8-sig","latin-1"]:
                    try:
                        return pd.read_csv(file_path, encoding=enc), None
                    except UnicodeDecodeError:
                        continue
            elif ext in {".xlsx",".xls"}:
                return pd.read_excel(file_path), None
            return None, f"[خطأ] صيغة غير مدعومة: {path.suffix}"
        except Exception as e:
            return None, f"[خطأ] فشل تحميل البيانات: {e}"


    def _inspect(self, df: pd.DataFrame, scope: list) -> list:
        """
        يُنتج قائمة مرتبة من المشاكل — كل مشكلة تحمل العملية الموصى بها.

        ترتيب التنفيذ مهم:
          1. الصفوف المكررة أولاً (تؤثر على الإحصاءات)
          2. الشذوذات ثانياً (قبل حساب المتوسطات)
          3. القيم المفقودة أخيراً (بعد إزالة الشذوذات)
        """
        issues = []

       
        dups = df.duplicated().sum()
        if dups > 0:
            issues.append({
                "col"   : "__all__",
                "type"  : "Error Correction",
                "op"    : "remove_duplicates",
                "reason": f"{dups} صف مكرر"
            })

        for col in scope:
            if col not in df.columns:
                continue
            s    = df[col]
            miss = s.isna().mean() * 100

          
            if miss > 70:
                issues.append({
                    "col"   : col,
                    "type"  : "Missing Imputation",
                    "op"    : "drop_high_missing",
                    "reason": f"مفقود {miss:.1f}% > 70% — يُقترح الحذف"
                })
                continue  

            # 3. شذوذات في الأعمدة الرقمية (Error Correction)
            if pd.api.types.is_numeric_dtype(s) and len(s.dropna()) > 10:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    out = ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).sum()
                    pct = out / len(s) * 100
                    if 0 < pct < 20:   
                        issues.append({
                            "col"   : col,
                            "type"  : "Error Correction",
                            "op"    : "remove_outliers",
                            "reason": f"{out} شذوذ ({pct:.1f}%) خارج 3×IQR"
                        })


            if s.dtype == "object":
                kws = ["date","time","year","month","day","تاريخ","وقت"]
                if any(k in col.lower() for k in kws):
                    issues.append({
                        "col"   : col,
                        "type"  : "Standardization",
                        "op"    : "fix_datetime",
                        "reason": "اسم العمود يشير إلى تاريخ"
                    })
                else:
                    # توحيد النص
                    issues.append({
                        "col"   : col,
                        "type"  : "Standardization",
                        "op"    : "standardize_text",
                        "reason": "توحيد النص: trim + lowercase"
                    })

            # 5. قيم مفقودة (Missing Imputation)
            if miss > 0:
                if pd.api.types.is_numeric_dtype(s):
                    clean = s.dropna()
                    skew  = abs(clean.skew()) if len(clean) > 3 else 0
                    if skew > 1:
                        op     = "fill_median"
                        detail = f"توزيع منحرف (skew={skew:.2f}) → وسيط"
                    else:
                        op     = "fill_mean"
                        detail = f"توزيع طبيعي (skew={skew:.2f}) → متوسط"
                else:
                    op     = "fill_mode"
                    detail = "عمود فئوي → المنوال"

                issues.append({
                    "col"   : col,
                    "type"  : "Missing Imputation",
                    "op"    : op,
                    "reason": f"مفقود {miss:.1f}% — {detail}"
                })

        return issues


    def _apply(self, df: pd.DataFrame, iss: dict) -> dict:
        col = iss["col"]
        op  = iss["op"]

        try:
            if op == "remove_duplicates":
                before = len(df)
                df = df.drop_duplicates().reset_index(drop=True)
                return {"ok": True, "df": df, "msg": f"حُذف {before - len(df)} صف مكرر"}

            if op == "remove_outliers":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr    = q3 - q1
                before = len(df)
                mask   = ~((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr))
                df     = df[mask].reset_index(drop=True)
                return {"ok": True, "df": df, "msg": f"حُذف {before - len(df)} شذوذ"}

            if op == "drop_high_missing":
                df = df.drop(columns=[col])
                return {"ok": True, "df": df, "msg": f"حُذف العمود '{col}'"}

            if op == "fix_datetime":
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                ok_count  = converted.notna().sum()
                df[col]   = converted
                return {"ok": True, "df": df, "msg": f"حُوِّل {ok_count} قيمة إلى datetime"}

            if op == "standardize_text":
                df[col] = df[col].astype(str).str.strip().str.lower()
                return {"ok": True, "df": df, "msg": "trim + lowercase مكتمل"}

            if op == "fill_mean":
                val   = df[col].mean()
                count = df[col].isna().sum()
                df[col] = df[col].fillna(round(val, 4))
                return {"ok": True, "df": df, "msg": f"ملء {count} قيمة بالمتوسط ({val:.4f})"}

            if op == "fill_median":
                val   = df[col].median()
                count = df[col].isna().sum()
                df[col] = df[col].fillna(val)
                return {"ok": True, "df": df, "msg": f"ملء {count} قيمة بالوسيط ({val})"}

            if op == "fill_mode":
                modes = df[col].mode()
                if len(modes) == 0:
                    return {"ok": False, "df": df, "msg": "لا توجد قيمة منوال"}
                count = df[col].isna().sum()
                df[col] = df[col].fillna(modes[0])
                return {"ok": True, "df": df, "msg": f"ملء {count} قيمة بالمنوال ({modes[0]!r})"}

            return {"ok": False, "df": df, "msg": f"عملية غير معروفة: {op}"}

        except Exception as e:
            return {"ok": False, "df": df, "msg": f"استثناء: {e}"}


    def _save(self, df: pd.DataFrame, original: str) -> str:
        p    = Path(original)
        out  = p.parent / f"{p.stem}_cleaned{p.suffix}"
        if p.suffix.lower() == ".csv":
            df.to_csv(out, index=False, encoding="utf-8-sig")
        else:
            df.to_excel(out, index=False)


        cache = os.path.join(tempfile.gettempdir(), f"agt_{p.stem}.pkl")
        df.to_pickle(cache)

        return str(out)
