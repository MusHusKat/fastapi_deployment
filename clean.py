import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df.head()

    # remove spaces from columns
    df.columns = [x.replace(" ", "") for x in df.columns]

    # remove spaces from each column
    for col in df:
        if df[col].dtype == "object":
            print(f"Removing spaces from column: '{col}'")
            df[col] = [x.replace(" ", "") for x in df[col]]

    # which columns have ?
    for col in df:
        subset_shape = df[df[col] == "?"].shape
        if subset_shape[0] > 0:
            df[col] = df[col].replace("?", "unknown")
            print(f"Replacing '?' with 'unknown' in  column: '{col}'")

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    clean_data(os.environ["RAW_DATA_PATH"], os.environ["CLEAN_DATA_PATH"])
