 csv_path = '/Users/fernando/Documents/Big data E IA/Semestre 1/Mineria de Datos/proyectoFinal/backend/Data/datasetfiltrosecadometalurgico.csv'
    inserter = DataInserter(csv_path, "FiltroSecadoMetalurgico4")
    inserter.run()
     print(df.head(10)) 
    print(df.columns)
  

   
    analisis.getHeatMap()
    analisis.histogramaDataFrame()
    analisis.dataBoxplot()
    analisis.dataBoxplotCol("Humedad")
    analisis.correlationmatrix()
  