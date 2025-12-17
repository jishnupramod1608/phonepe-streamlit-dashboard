import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from io import BytesIO

# =========================================================
# STEP 1: PHONEPE-AWARE RULE ENGINE (PRIMARY)
# =========================================================
def rule_based_category(text):
    t = str(text).lower()

    if "received from" in t or "credited" in t or "salary" in t:
        return "Income"

    if any(x in t for x in ["greenmart", "grocery", "bigbasket", "dmart", "supermarket"]):
        return "Groceries"

    if any(x in t for x in ["utility", "electric", "electricity", "water", "gas", "airtel", "jio", "recharge"]):
        return "Utilities"

    if any(x in t for x in ["metro", "transport", "irctc", "bus", "uber", "ola", "rapido"]):
        return "Transport"

    if any(x in t for x in ["traders", "store", "mart", "amazon", "flipkart", "myntra"]):
        return "Shopping"

    if any(x in t for x in ["zomato", "swiggy", "restaurant", "food", "cafe"]):
        return "Food"

    if any(x in t for x in ["netflix", "prime", "hotstar", "movie"]):
        return "Entertainment"

    if any(x in t for x in ["emi", "loan", "credit card"]):
        return "EMI / Loans"

    return "Other"


# =========================================================
# STEP 2: TRAIN ML MODEL (ONLY IF DATA IS GOOD)
# =========================================================
def train_category_model_from_file(df, details_col):
    texts = df[details_col].astype(str)
    labels = texts.apply(rule_based_category)

    train_df = pd.DataFrame({"text": texts, "label": labels})
    train_df = train_df[train_df["label"] != "Other"]

    if train_df["label"].nunique() < 2:
        return None, None

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["label"]

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    return vectorizer, model


# =========================================================
# STEP 3: APPLY ML WITH RULE FALLBACK
# =========================================================
def categorize_transactions(df, details_col):
    df["Category"] = df[details_col].apply(rule_based_category)

    vectorizer, model = train_category_model_from_file(df, details_col)
    if vectorizer is None:
        return df

    X = vectorizer.transform(df[details_col].astype(str))
    preds = model.predict(X)
    probs = model.predict_proba(X).max(axis=1)

    final = []
    for text, pred, prob in zip(df[details_col], preds, probs):
        final.append(pred if prob >= 0.6 else rule_based_category(text))

    df["Category"] = final
    return df


# =========================================================
# STEP 4: CORRECT INCOME / EXPENSE CALCULATION
# =========================================================
def calculate_financials(df):
    df_tmp = df.copy()
    df_tmp.columns = [c.lower().strip() for c in df_tmp.columns]

    amount_col = next(c for c in df_tmp.columns if "amount" in c)
    details_col = next(c for c in df_tmp.columns if "transaction details" in c)

    df_tmp[amount_col] = pd.to_numeric(df_tmp[amount_col], errors="coerce").fillna(0)

    income_mask = df_tmp[details_col].str.contains(
        "received from|credited|salary", case=False, na=False
    )

    expense_mask = df_tmp[details_col].str.contains(
        "paid to|debit|upi", case=False, na=False
    )

    income = df_tmp.loc[income_mask, amount_col].sum()
    expense = df_tmp.loc[expense_mask, amount_col].sum()

    savings_rate = ((income - expense) / income * 100) if income > 0 else 0
    return round(income, 2), round(expense, 2), round(savings_rate, 2)


# =========================================================
# STEP 5: CONTENT SUGGESTIONS ENGINE
# =========================================================
def generate_content_suggestions(df, income, expense, savings_rate):
    suggestions = []

    if savings_rate < 10:
        suggestions.append("⚠️ Your savings rate is very low. Track daily expenses to identify money leaks.")
        suggestions.append("📉 Apply the 50-30-20 budgeting rule.")
    elif savings_rate < 30:
        suggestions.append("🙂 Your savings are moderate. Try increasing them to at least 30%.")
        suggestions.append("💡 Reduce non-essential spending.")
    else:
        suggestions.append("🎉 Great job! You are saving well.")
        suggestions.append("📈 Consider investing surplus money via SIPs or mutual funds.")

    amount_col = next(c for c in df.columns if "amount" in c.lower())
    expense_df = df[df["Category"] != "Income"]

    if not expense_df.empty:
        top_category = expense_df.groupby("Category")[amount_col].sum().idxmax()

        tips = {
            "Food": "🍔 High food spending detected. Reduce online orders and cook more at home.",
            "Shopping": "🛍️ Shopping is high. Set a monthly shopping budget.",
            "Utilities": "💡 Utilities are costly. Look for energy-saving options.",
            "Transport": "🚗 Transport costs are high. Try public transport or pooling.",
            "EMI / Loans": "🏦 EMI burden is high. Prepay high-interest loans if possible."
        }

        suggestions.append(tips.get(
            top_category,
            f"📊 Your highest spending category is **{top_category}**. Review it carefully."
        ))

    return suggestions


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config("FinanceWise Mentor", "💰", layout="wide")
st.title("FinanceWise Mentor Dashboard")

uploaded_file = st.file_uploader("Upload PhonePe CSV / Excel", ["csv", "xlsx"])

if uploaded_file:
    is_csv = uploaded_file.name.endswith(".csv")
    df = pd.read_csv(uploaded_file) if is_csv else pd.read_excel(uploaded_file)

    details_col = next(c for c in df.columns if "transaction details" in c.lower())

    df = categorize_transactions(df, details_col)
    income, expense, savings_rate = calculate_financials(df)
   
# ================== STEP 2: SEND DATA TO DASHBOARD ==================
st.session_state["processed_df"] = df
st.session_state["income"] = income
st.session_state["expense"] = expense
st.session_state["savings_rate"] = savings_rate

st.markdown("---")

if st.button("Go to Analytics Dashboard"):
    st.switch_page("app1.py")

    st.subheader("📊 Financial Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Income", f"₹{income}")
    c2.metric("Expense", f"₹{expense}")
    c3.metric("Savings Rate", f"{savings_rate}%")

    st.markdown("---")
    st.subheader("📂 Category-wise Spending (₹)")
    amount_col = next(c for c in df.columns if "amount" in c.lower())
    st.bar_chart(df[df["Category"] != "Income"].groupby("Category")[amount_col].sum())

    # ================= CONTENT SUGGESTIONS =================
    st.markdown("---")
    st.subheader("🧠 Personalized Content Suggestions")

    suggestions = generate_content_suggestions(df, income, expense, savings_rate)
    for s in suggestions:
        st.markdown(f"- {s}")

    # ================= DOWNLOAD =================
    st.markdown("---")
    st.subheader("⬇️ Download Categorized File")

    if is_csv:
        st.download_button(
            "Download CSV with Categories",
            df.to_csv(index=False).encode("utf-8"),
            "transactions_with_categories.csv",
            "text/csv"
        )
    else:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)

        st.download_button(
            "Download Excel with Categories",
            buffer,
            "transactions_with_categories.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Upload your PhonePe transaction file to continue")

