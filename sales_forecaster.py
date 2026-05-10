from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class SalesForecasterTool:
    DATE_KEYWORDS = [
        "date", "time", "day", "month",
        "order date", "transaction date", "invoice date",
        "created_at", "sold_at"
    ]

    SALES_KEYWORDS = [
        "sales", "revenue", "amount", "total", "price",
        "total spent", "total_spent", "subtotal", "income"
    ]

    def run(
        self,
        file_path: str,
        date_column: str | None = None,
        sales_column: str | None = None,
        months_to_predict: int = 3
    ) -> str:
        path = Path(file_path)

        if not path.exists():
            return f"[خطأ] الملف غير موجود: {file_path}"

        df = pd.read_csv(path)

        if df.empty:
            return "[خطأ] الملف فارغ."

        date_column = date_column or self._detect_date_column(df)
        sales_column = sales_column or self._detect_sales_column(df)

        if not date_column:
            return (
                "[خطأ] لم أستطع اكتشاف عمود التاريخ تلقائيًا.\n"
                "مرّر date_column يدويًا."
            )

        if not sales_column:
            return (
                "[خطأ] لم أستطع اكتشاف عمود المبيعات تلقائيًا.\n"
                "مرّر sales_column يدويًا."
            )

        data = df[[date_column, sales_column]].copy()

        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data[sales_column] = (
            data[sales_column]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        data[sales_column] = pd.to_numeric(data[sales_column], errors="coerce")

        before_rows = len(data)
        data = data.dropna(subset=[date_column, sales_column])
        after_rows = len(data)

        if data.empty:
            return "[خطأ] بعد تنظيف التاريخ والمبيعات لم تبقَ بيانات صالحة."

        data["month"] = data[date_column].dt.to_period("M").dt.to_timestamp()

        monthly = (
            data.groupby("month")[sales_column]
            .sum()
            .reset_index()
            .sort_values("month")
        )

        if len(monthly) < 3:
            return (
                "[تحذير] البيانات تحتوي على أقل من 3 أشهر.\n"
                "يمكن عرض المبيعات الشهرية، لكن التوقع لن يكون موثوقًا.\n\n"
                + monthly.to_string(index=False)
            )

        monthly["month_index"] = np.arange(len(monthly))

        X = monthly[["month_index"]]
        y = monthly[sales_column]

        model = LinearRegression()
        model.fit(X, y)

        future_indexes = np.arange(
            len(monthly),
            len(monthly) + months_to_predict
        ).reshape(-1, 1)

        predictions = model.predict(future_indexes)
        predictions = np.maximum(predictions, 0)

        last_months = monthly.tail(3)
        first_last = last_months[sales_column].iloc[0]
        final_last = last_months[sales_column].iloc[-1]

        if first_last != 0:
            growth_rate = ((final_last - first_last) / first_last) * 100
        else:
            growth_rate = 0

        trend = "تصاعدي" if growth_rate > 5 else "تنازلي" if growth_rate < -5 else "مستقر"

        report = []
        report.append("=" * 60)
        report.append("تقرير تحليل وتوقع المبيعات")
        report.append("=" * 60)
        report.append(f"الملف: {file_path}")
        report.append(f"عمود التاريخ المكتشف: {date_column}")
        report.append(f"عمود المبيعات المكتشف: {sales_column}")
        report.append(f"عدد الصفوف قبل التنظيف: {before_rows}")
        report.append(f"عدد الصفوف الصالحة بعد التنظيف: {after_rows}")
        report.append("")
        report.append("آخر 3 أشهر:")
        for _, row in last_months.iterrows():
            report.append(
                f"- {row['month'].strftime('%Y-%m')}: "
                f"{row[sales_column]:,.2f}"
            )

        report.append("")
        report.append(f"اتجاه آخر 3 أشهر: {trend}")
        report.append(f"نسبة التغير التقريبية: {growth_rate:.2f}%")
        report.append("")
        report.append(f"توقع المبيعات للأشهر القادمة ({months_to_predict}):")

        last_date = monthly["month"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=months_to_predict,
            freq="MS"
        )

        for date, pred in zip(future_dates, predictions):
            report.append(f"- {date.strftime('%Y-%m')}: {pred:,.2f}")

        report.append("")
        report.append("ملاحظة:")
        report.append(
            "هذا توقع بسيط باستخدام Linear Regression على المبيعات الشهرية. "
            "مناسب كبداية عامة لأي بيانات مبيعات، ويمكن تطويره لاحقًا."
        )
        report.append("=" * 60)

        return "\n".join(report)

    def _detect_date_column(self, df: pd.DataFrame) -> str | None:
        columns = list(df.columns)

        for col in columns:
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in self.DATE_KEYWORDS):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    return col

        for col in columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                return col

        return None

    def _detect_sales_column(self, df: pd.DataFrame) -> str | None:
        columns = list(df.columns)

        candidates = []

        for col in columns:
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in self.SALES_KEYWORDS):
                numeric = pd.to_numeric(
                    df[col].astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip(),
                    errors="coerce"
                )
                valid_ratio = numeric.notna().mean()
                if valid_ratio > 0.5:
                    candidates.append((col, valid_ratio, numeric.mean()))

        if candidates:
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return candidates[0][0]

        numeric_candidates = []
        for col in columns:
            numeric = pd.to_numeric(
                df[col].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip(),
                errors="coerce"
            )
            valid_ratio = numeric.notna().mean()
            if valid_ratio > 0.7 and numeric.mean() > 0:
                numeric_candidates.append((col, numeric.mean()))

        if numeric_candidates:
            numeric_candidates.sort(key=lambda x: x[1], reverse=True)
            return numeric_candidates[0][0]

        return None