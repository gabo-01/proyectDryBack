import os
import sys
import pandas as pd

from dataBase.DataInserter import DataInserter
from ML.configuration.ConfigDataFrame import ConfigDataFrame
from ML.clean.CleanData import CleanData
from ML.ANALISIS.analisis import analisisDataFrame
 
 
from ML.ModelTraining.ModeloHumedad import ModeloHumedad
def main():
    
    # Configuración de la ruta del archivo CSV
    #csv_path = '/Users/fernando/Documents/Big data E IA/Semestre 1/Mineria de Datos/proyectoFinal/backend/Data/datasetfiltrosecadometalurgico.csv'
    #inserter = DataInserter(csv_path, "FiltroSecadoMetalurgico4")
    #inserter.run()
    dataFrame = ConfigDataFrame()
    df = dataFrame.get_dataframe("FiltroSecadoMetalurgico3")
    print(df.head(10)) 
    print(df.columns)
  

    #analisis= analisisDataFrame(df)
    #analisis.getHeatMap()
    #analisis.histogramaDataFrame()
    #analisis.dataBoxplot()
    #analisis.dataBoxplotCol("Humedad")
    #analisis.correlationmatrix()
    #analisis.outliers()
    
    
   
  
    #cleaner = CleanData(df, "FiltroSecadoMetalurgicoLimpio")
    #cleaner.clean_data_iterative()
    #df_limpio = cleaner.dataframe
    #print("🧼 DataFrame limpio:")
    
    #print(f"Total de registros: {len(df_limpio)}")
    #cleaner.insert_data()
    #print(df.head(10))
    modelo = ModeloHumedad(df)
    modelo.verificar_datos()
    modelo._preparar_target()  
    modelo.limpiar_datos()
    modelo.seleccionar_caracteristicas()
    modelo.preparar_datos()

     
    print("Entrenando el modelo de regresión...")
    modelo.entrenar_regresion(optimizar=True)  # Entrena el modelo de regresión
    
    print("Entrenando el modelo de clasificación...")
    modelo.entrenar_clasificacion(optimizar=True)  # Entrena el modelo de clasificación
    
    # Evaluar los modelos
    modelo.evaluar_regresion()  # Evaluar el modelo de regresión
    modelo.evaluar_clasificacion()
    ruta="/Users/fernando/Documents/Big data E IA/Semestre 1/Mineria de Datos/"
    modelo.guardar_modelos(ruta_regresion=ruta + "modelo_humedad_regresion.pkl", ruta_clasificacion=ruta+"modelo_humedad_clasificacion.pkl")  # Guardar los modelos entrenados
  


if __name__ == "__main__":
    main()
