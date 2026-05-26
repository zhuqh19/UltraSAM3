import pandas as pd
import sys

file_path = r"C:\Users\zhuqh\Desktop\sam3\datasets\心脏超声数据集\18.心脏超声(EchoCP)\archive\echoCP_diagnosis_label.xlsx"

try:
    df = pd.read_excel(file_path)
    print(df.head())
    print("-" * 20)
    print(df.columns)
except Exception as e:
    print(f"Error: {e}")
