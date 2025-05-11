# backend/database/data_inserter.py
import pandas as pd
import os
import sys
 
from  .MySQLConnector  import MySQLConnector
class DataInserter:
    def __init__(self, filepath, table_name):
        self.filepath = filepath
        self.table_name = table_name
        self.db = MySQLConnector()  # Ya usa el config desde dentro

    def read_data(self):
        print(f"📄 Leyendo archivo: {self.filepath}")
        df = pd.read_csv(self.filepath)

        # Elimina columnas vacías, si las hay
        df = df.dropna(axis=1, how='all')

        print(f"🔍 Columnas detectadas: {df.columns.tolist()}")
        print(df.head(3))  # Muestra las primeras filas para verificar
        return df

    def run(self):
        try:
            df = self.read_data()
            self.db.crear_tabla(df, self.table_name)  # ahora pasa el DataFrame
            self.db.insertar_dataframe(df, self.table_name)
            print("✅ Datos insertados correctamente en la base de datos.")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.db.cerrar()