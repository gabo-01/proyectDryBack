import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class analisisDataFrame:
    """
    Clase para analizar un DataFrame de pandas.
    """

    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame.

        :param df: DataFrame de pandas a analizar.
        """
        self.df = df

    def getHeatMap(self):
        """
        Genera un mapa de calor de valores nulos para el DataFrame.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

    def histogramaDataFrame(self):
        """
        Genera histogramas para cada columna numérica del DataFrame.
        """
        self.df.hist(bins=50, figsize=(29, 15))
        plt.tight_layout()
        plt.show()

    def dataBoxplot(self):
        """
        Genera gráficos de caja (boxplot) separados por columna pero todos en una sola figura.
        """
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])

        num_cols = numeric_df.shape[1]
        fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(12, num_cols * 3))  # 3 de alto por plot

        for i, column in enumerate(numeric_df.columns):
            sns.boxplot(data=numeric_df, x=column, ax=axes[i])
            axes[i].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

    def dataBoxplotCol(self, column_name):
        """
        Genera un gráfico de caja (boxplot) para una sola columna especificada.

        :param column_name: Nombre de la columna a graficar.
        """
        if column_name not in self.df.columns:
            print(f"❌ La columna '{column_name}' no existe en el DataFrame.")
            return

        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            print(f"❌ La columna '{column_name}' no es numérica.")
            return

        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df[column_name])
        plt.title(f'Boxplot de {column_name}')
        plt.xlabel(column_name)
        plt.tight_layout()
        plt.show()

    def correlationmatrix(self):
        """
        Genera una matriz de correlación para el DataFrame.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.show()
    
    def outliers(self):
        """
        Devuelve los outliers del DataFrame usando el método del IQR.
        """
        df_numeric = self.df.select_dtypes(include=['float64'])  # solo columnas numéricas

        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.0 * IQR  # Ajusta el factor multiplicador
        upper_bound = Q3 + 1.0 * IQR

        # Filtrar filas que contienen al menos un outlier
        outliers = (df_numeric < lower_bound) | (df_numeric > upper_bound)
        outliers_count = outliers.sum()
        
        # Filas que tienen outliers
        filas_outliers = df_numeric[outliers.any(axis=1)]
        
        return outliers_count, filas_outliers
    def feature_importance_plot(self, model, feature_names=None):
        """
        Genera una gráfica de importancia de características (feature importance) para modelos con el atributo `feature_importances_`.

        :param model: Modelo entrenado que tenga el atributo `feature_importances_`.
        :param feature_names: Lista de nombres de las características. Si no se proporciona, se intentará usar self.df.columns.
        """
        if not hasattr(model, 'feature_importances_'):
            print("❌ El modelo no tiene el atributo `feature_importances_`.")
            return

        importances = model.feature_importances_

        if feature_names is None:
            feature_names = self.df.columns

        if len(importances) != len(feature_names):
            print("❌ La cantidad de importancias no coincide con la cantidad de nombres de características.")
            print(f"🔎 Longitud de importances: {len(importances)}, longitud de feature_names: {len(feature_names)}")
            return

        # Crear DataFrame para graficar ordenado
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Importancia de las Características')
        plt.tight_layout()
        plt.show()
    
    
 