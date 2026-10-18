
def csv_bytes(df,user_name):
    out=df.copy()
    out.insert(1,"Gardener",user_name)
    if "ID" in out.columns: out=out.rename(columns={"ID":"Scan ID"})
    return out.to_csv(index=False).encode("utf-8-sig")
