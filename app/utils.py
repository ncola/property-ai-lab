import pandas as pd

def fmt_money(x):
    if x is None or pd.isna(x):
        return "brak informacji"
    try:
        return f"{float(x):,.0f} zł".replace(",", " ")
    except Exception:
        return str(x)

def cut_description(text, max_length=300):
    text = text.strip(" ")
    if len(text) <= max_length:
        return text
    
    cutoff = text.rfind(" ", 0, max_length)
    if cutoff == -1:
        return f"{text[:max_length]}..."
    return f"{text[:cutoff]}..."

def calculate_price(df):
    area = df['area']
    predicted_price_per_m2 = df['predicted_price_per_m2']

    return (area*predicted_price_per_m2).astype(int)

def is_deal(df, threshold = 0.2):
    df = df.copy()

    required_cols = ["price", "predicted_price"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    price_diff_ratio = (df["predicted_price"] - df["price"]) / df["predicted_price"]

    deal = price_diff_ratio >= threshold

    return price_diff_ratio, deal
