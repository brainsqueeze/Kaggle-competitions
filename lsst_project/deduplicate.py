import pandas as pd
import os

root = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(root + "/output/submission.csv")
data = data.drop_duplicates(subset="object_id", keep="first")
data.to_csv(root + "/output/submission_dedupe.csv")
