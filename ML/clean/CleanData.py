import pandas as pd
from dataBase.MySQLConnector import MySQLConnector

class CleanData:
    def __init__(self, dataframe, tableName):
        self.tableName = tableName
        self.dataframe = dataframe
        self.db = MySQLConnector()

    def clean_data(self):
        """Limpia el DataFrame eliminando outliers y rellenando valores nulos con la media."""
        df_numeric = self.dataframe.select_dtypes(include=['float64', 'int64'])

        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1

        # Ajusta los límites para ser más estrictos
        lower_bound = Q1 - 1.0 * IQR  # Ajusta el factor multiplicador
        upper_bound = Q3 + 1.0 * IQR

        # Filtra filas que NO tienen outliers
        mask = ~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)
        original_shape = self.dataframe.shape

        self.dataframe = self.dataframe[mask]

        print(f"🧹 Outliers eliminados: {original_shape[0] - self.dataframe.shape[0]} filas eliminadas")

        # Rellenar nulos
        self.dataframe = self.dataframe.fillna(self.dataframe.mean(numeric_only=True))
        if self.dataframe.isnull().sum().sum() == 0:
            print("🧼 Nulos rellenados con la media")

    def clean_data_iterative(self, max_iter=10):
        iter_count = 0
        while True:
            before_shape = self.dataframe.shape[0]
            self.clean_data()  # tu método actual de limpieza
            after_shape = self.dataframe.shape[0]
            print(f"🧹 Iteración {iter_count + 1}: {before_shape - after_shape} filas eliminadas")
            iter_count += 1
            if before_shape == after_shape or iter_count >= max_iter:
                break
    
    def insert_data(self):
        try:
            self.db.crear_tabla(self.dataframe,self.tableName)

            
            if 'id' in self.dataframe.columns:
                self.dataframe = self.dataframe.drop(columns=['id'])

            self.db.insertar_dataframe(self.dataframe, self.tableName)
            print(f"✅ Datos insertados correctamente en la tabla '{self.tableName}'")
        except Exception as e:
            print(f"❌ Error al insertar: {e}")
        finally:
            self.db.cerrar()

 