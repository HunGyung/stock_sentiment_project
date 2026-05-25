import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.preprocessor import clean_news_text

def explore():
    # load data
    df = pd.read_csv("data/finance_data.csv")
    print(f"전체 데이터 수 : {len(df)}")

    # check distribution of label
    label_counts = df['labels'].value_counts()
    print("\n[레이블 분포]")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f" - {label}: {count}개 ({percentage:.2f}%)")

    print("\n[전처리 전/후 비교]")
    sample_sentence = df.iloc[3]['kor_sentence']
    print(f" - 전처리 전: {sample_sentence}")
    print(f" - 전처리 후: {clean_news_text(sample_sentence)}")

if __name__ == "__main__":
    explore()