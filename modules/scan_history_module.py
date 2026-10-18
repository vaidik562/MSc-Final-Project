
import pandas as pd
from modules.database_module import get_scans

COLUMNS=["ID","Date","Plant","Problem","Severity","Confidence","Smart Watering","Medicine","Cost","Detection Method"]

def history_df(user_id):
    return pd.DataFrame(get_scans(user_id),columns=COLUMNS)
