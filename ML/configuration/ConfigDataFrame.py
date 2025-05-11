import pandas as pd
from dataBase.MySQLConnector import MySQLConnector

class ConfigDataFrame:
    def __init__(self):
        self.db = MySQLConnector()

    def get_dataframe(self, table_name):
        rows, columns = self.db.select(table_name)
        df = pd.DataFrame(rows, columns=columns)
        print("📦 DataFrame creado con éxito")
        return df

