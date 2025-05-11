 
import mysql.connector
import pandas as pd
from .db_config  import db_config

class MySQLConnector:
    def __init__(self,config=db_config):
        self.conn = mysql.connector.connect(
           host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
        self.cursor = self.conn.cursor()

    def crear_tabla(self, df, table_name):
        def infer_mysql_type(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                return "INT"
            elif pd.api.types.is_float_dtype(dtype):
                return "FLOAT"
            elif pd.api.types.is_bool_dtype(dtype):
                return "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                return "DATETIME"
            else:
                return "VARCHAR(255)"

        columns_sql = []
        for col in df.columns:
            if col.lower() == 'id':
                continue  # Evita agregar la columna 'id' si ya existe en el DataFrame

            mysql_type = infer_mysql_type(df[col].dtype)
            col_name = col.replace(" ", "_")
            columns_sql.append(f"`{col_name}` {mysql_type}")

        columns_sql_str = ",\n    ".join(columns_sql)

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY{',' if columns_sql_str else ''}
            {columns_sql_str}
        )
        """
        self.cursor.execute(create_sql)
    def insertar_dataframe(self, df, table_name):
        columns = df.columns.tolist()
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)

        query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        """
        for _, row in df.iterrows():
            values = [None if pd.isna(val) else val for val in row]
            self.cursor.execute(query, values)
        self.conn.commit()
    
    def select(self, table_name):
        query = f"SELECT * FROM {table_name}"
        
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return rows, columns

    def cerrar(self):
            self.cursor.close()
            self.conn.close()