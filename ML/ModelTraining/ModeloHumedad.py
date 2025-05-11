import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


class ModeloHumedad:
    """
    Clase para modelado de predicción de HumedadFinal_pct
    Ofrece dos tipos de modelos:
    - Un modelo de regresión para predecir el valor exacto
    - Un modelo de clasificación para predecir si la humedad será alta o baja
    """
    
    def __init__(self, df):
        """
        Inicializa el modelo con el DataFrame proporcionado
        
        Parámetros:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos, debe incluir la columna 'HumedadFinal_pct'
        """
        self.df = df.copy()
        self.verificar_datos()
        
        # Modelos
        self.modelo_regresion = None
        self.modelo_clasificacion = None
        
        # Umbrales y métricas
        self.umbral_clasificacion = 0.4 # Umbral para clasificar entre humedad alta/baja
        self.scaler = StandardScaler()
        
        # Preparar variables objetivo
        if 'HumedadFinal_pct' in self.df.columns:
            self._preparar_target()
    
    def verificar_datos(self):
        """Verifica la integridad de los datos"""
        # Verificar columna target
        if 'HumedadFinal_pct' not in self.df.columns:
            raise ValueError("El DataFrame debe contener la columna 'HumedadFinal_pct'")
        
        # Verificar variables importantes
        variables_importantes = [
            'HumedadInicial_pct', 'ÁreaFiltro_cm2', 'HumedadAmbiental_pct', 
            'DensidadMaterial_g_cm3'
        ]
        
        for var in variables_importantes:
            if var not in self.df.columns:
                print(f"Advertencia: Variable importante '{var}' no encontrada")
        
        # Verificar valores nulos
        nulos = self.df.isnull().sum()
        if nulos.sum() > 0:
            print(f"Advertencia: Se encontraron {nulos.sum()} valores nulos en el dataset")
            print(nulos[nulos > 0])
    
    def _preparar_target(self):
        
        """Prepara la variable objetivo para clasificación"""
        # Crear variable categórica basada en umbral
        self.df['HumedadFinal_Clase'] = (self.df['HumedadFinal_pct'] > self.umbral_clasificacion).astype(int)
        
        # Verificar distribución de clases
        conteo_clases = self.df['HumedadFinal_Clase'].value_counts()
        print(f"Distribución de clases (umbral={self.umbral_clasificacion}):")
        print(f"Clase 0 (Humedad baja): {conteo_clases.get(0, 0)} muestras")
        print(f"Clase 1 (Humedad alta): {conteo_clases.get(1, 0)} muestras")
    

    def seleccionar_caracteristicas(self, metodo='correlacion', n_features=10):
        """
        Selecciona las mejores características para el modelo basándose en la correlación con la variable objetivo.
        
        Parámetros:
        -----------
        metodo : str
            Método de selección ('correlacion' actualmente implementado)
        n_features : int
            Número de características a seleccionar
        """
        if metodo == 'correlacion':
            
            df_numerico = self.df.select_dtypes(include=[np.number])
            
             
            if 'HumedadFinal_pct' not in df_numerico.columns:
                raise ValueError("'HumedadFinal_pct' no es una columna numérica o no está presente.")
            
          
            corr = df_numerico.corr()['HumedadFinal_pct'].abs().sort_values(ascending=False)
            
             
            corr = corr.drop('HumedadFinal_pct', errors='ignore')
            
             
            return list(corr.head(n_features).index)

        else:
            raise NotImplementedError(f"Método de selección '{metodo}' no está implementado.")

    
    def limpiar_datos(self):
        """Convierte o elimina columnas no numéricas innecesarias."""
        # Convertir columnas categóricas o de texto si las vas a usar
        if 'EquipoID' in self.df.columns:
            self.df['EquipoID'] = self.df['EquipoID'].astype('category').cat.codes
        if 'HoraInicio' in self.df.columns:
            self.df['HoraInicio'] = pd.to_datetime(self.df['HoraInicio'], errors='coerce')
            self.df['HoraInicio_ts'] = self.df['HoraInicio'].astype('int64')  # usar solo si tiene valor
        if 'Observaciones' in self.df.columns:
            self.df['Observaciones'] = self.df['Observaciones'].fillna('')  # o eliminarla


    def preparar_datos(self, caracteristicas=None, test_size=0.2, random_state=42):
        """
        Prepara los datos para entrenamiento/prueba
        
        Parámetros:
        -----------
        caracteristicas : list
            Lista de columnas a usar como features (si None, usa selección automática)
        test_size : float
            Proporción de datos para prueba (0.0-1.0)
        random_state : int
            Semilla para reproducibilidad
        """
        # Seleccionar características si no se especifican
        if caracteristicas is None:
            caracteristicas = self.seleccionar_caracteristicas()
        
       
        self.X_nombres = caracteristicas
        
       
        X = self.df[caracteristicas]
        y_reg = self.df['HumedadFinal_pct']
        y_cls = self.df['HumedadFinal_Clase']
        
       
        self.X_train, self.X_test, self.y_train_reg, self.y_test_reg = train_test_split(
            X, y_reg, test_size=test_size, random_state=random_state
        )
        
        _, _, self.y_train_cls, self.y_test_cls = train_test_split(
            X, y_cls, test_size=test_size, random_state=random_state
        )
        
        # Escalar datos
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Datos preparados: {self.X_train.shape[0]} muestras de entrenamiento, {self.X_test.shape[0]} de prueba")
        print(f"Características utilizadas ({len(caracteristicas)}): {', '.join(caracteristicas)}")
    
    def entrenar_regresion(self, optimizar=False):
        """
        Entrena el modelo de regresión
        
        Parámetros:
        -----------
        optimizar : bool
            Si True, realiza búsqueda de hiperparámetros
        """
        if not hasattr(self, 'X_train'):
            self.preparar_datos()
        
        print("Entrenando modelo de regresión...")
        
        if optimizar:
            # GridSearch para optimización de hiperparámetros
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train_reg)
            self.modelo_regresion = grid_search.best_estimator_
            print(f"Mejores parámetros: {grid_search.best_params_}")
        else:
            # Modelo con parámetros por defecto
            self.modelo_regresion = RandomForestRegressor(
                n_estimators=200, 
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
            self.modelo_regresion.fit(self.X_train_scaled, self.y_train_reg)
        
        # Evaluar modelo
        self.evaluar_regresion()
        
        # Calcular importancia de características
        self._calcular_importancia_caracteristicas()
    
    def entrenar_clasificacion(self, optimizar=False):
        """
        Entrena el modelo de clasificación
        
        Parámetros:
        -----------
        optimizar : bool
            Si True, realiza búsqueda de hiperparámetros
        """
        if not hasattr(self, 'X_train'):
            self.preparar_datos()
        
        print("Entrenando modelo de clasificación...")
        
        if optimizar:
            # GridSearch para optimización de hiperparámetros
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            
            base_model = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, 
                scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train_cls)
            self.modelo_clasificacion = grid_search.best_estimator_
            print(f"Mejores parámetros: {grid_search.best_params_}")
        else:
            # Modelo con parámetros por defecto
            self.modelo_clasificacion = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.modelo_clasificacion.fit(self.X_train_scaled, self.y_train_cls)
        
        # Evaluar modelo
        self.evaluar_clasificacion()
    
    def evaluar_regresion(self):
        """Evalúa el rendimiento del modelo de regresión"""
        if self.modelo_regresion is None:
            print("Modelo de regresión no entrenado aún")
            return
        
        # Predicciones
        y_pred = self.modelo_regresion.predict(self.X_test_scaled)
        
        # Métricas
        mse = mean_squared_error(self.y_test_reg, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test_reg, y_pred)
        
        print("\nEvaluación del modelo de regresión:")
        print(f"Error cuadrático medio (MSE): {mse:.4f}")
        print(f"Raíz del error cuadrático medio (RMSE): {rmse:.4f}")
        print(f"Coeficiente de determinación (R²): {r2:.4f}")
        
        # Graficar predicciones vs valores reales
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test_reg, y_pred, alpha=0.5)
        plt.plot([0, max(self.y_test_reg)], [0, max(self.y_test_reg)], 'r--')
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title('Valores reales vs predicciones (Regresión)')
        plt.tight_layout()
        plt.show()
    
    def evaluar_clasificacion(self):
        """Evalúa el rendimiento del modelo de clasificación"""
        if self.modelo_clasificacion is None:
            print("Modelo de clasificación no entrenado aún")
            return
        
        # Predicciones
        y_pred = self.modelo_clasificacion.predict(self.X_test_scaled)
        
        # Métricas
        accuracy = accuracy_score(self.y_test_cls, y_pred)
        report = classification_report(self.y_test_cls, y_pred)
        
        print("\nEvaluación del modelo de clasificación:")
        print(f"Exactitud (Accuracy): {accuracy:.4f}")
        print("\nInforme de clasificación:")
        print(report)
    
    def _calcular_importancia_caracteristicas(self):
        """Calcula y muestra la importancia de características para el modelo de regresión"""
        if self.modelo_regresion is None:
            return
        
        # Importancia de características
        importancias = pd.Series(
            self.modelo_regresion.feature_importances_,
            index=self.X_nombres
        ).sort_values(ascending=False)
        
        # Gráfico de importancia
        plt.figure(figsize=(10, 6))
        importancias.plot(kind='bar')
        plt.title('Importancia de características')
        plt.ylabel('Importancia')
        plt.tight_layout()
        plt.show()
        
        print("\nImportancia de características:")
        for nombre, importancia in importancias.items():
            print(f"{nombre}: {importancia:.4f}")
    
    def predecir(self, nuevos_datos, tipo='regresion'):
        """
        Realiza predicciones para nuevos datos
        
        Parámetros:
        -----------
        nuevos_datos : pandas.DataFrame o array-like
            Nuevos datos para los que hacer predicciones
        tipo : str
            Tipo de predicción ('regresion' o 'clasificacion')
        
        Retorna:
        --------
        array : Predicciones
        """
        # Verificar que tenemos el modelo entrenado
        if tipo == 'regresion' and self.modelo_regresion is None:
            raise ValueError("El modelo de regresión no está entrenado")
        elif tipo == 'clasificacion' and self.modelo_clasificacion is None:
            raise ValueError("El modelo de clasificación no está entrenado")
        
        # Preparar datos
        if isinstance(nuevos_datos, pd.DataFrame):
            # Asegurar que tenemos las columnas necesarias
            columnas_faltantes = set(self.X_nombres) - set(nuevos_datos.columns)
            if columnas_faltantes:
                raise ValueError(f"Faltan columnas en los datos: {columnas_faltantes}")
            
            # Seleccionar solo las columnas necesarias y en el orden correcto
            X = nuevos_datos[self.X_nombres]
        else:
            # Convertir array a DataFrame
            X = pd.DataFrame(nuevos_datos, columns=self.X_nombres)
        
        # Escalar datos
        X_scaled = self.scaler.transform(X)
        
        # Realizar predicciones
        if tipo == 'regresion':
            return self.modelo_regresion.predict(X_scaled)
        else:
            return self.modelo_clasificacion.predict(X_scaled)
    
    def predecir_probabilidad(self, nuevos_datos):
        """
        Realiza predicciones de probabilidad para el modelo de clasificación
        
        Parámetros:
        -----------
        nuevos_datos : pandas.DataFrame o array-like
            Nuevos datos para los que hacer predicciones
        
        Retorna:
        --------
        array : Probabilidades de clase
        """
        if self.modelo_clasificacion is None:
            raise ValueError("El modelo de clasificación no está entrenado")
        
        # Preparar datos
        if isinstance(nuevos_datos, pd.DataFrame):
            X = nuevos_datos[self.X_nombres]
        else:
            X = pd.DataFrame(nuevos_datos, columns=self.X_nombres)
        
        X_scaled = self.scaler.transform(X)
        
        return self.modelo_clasificacion.predict_proba(X_scaled)
    
    def guardar_modelos(self, ruta_regresion='modelo_humedad_regresion.pkl', 
                       ruta_clasificacion='modelo_humedad_clasificacion.pkl'):
        """
        Guarda los modelos entrenados
        
        Parámetros:
        -----------
        ruta_regresion : str
            Ruta para guardar el modelo de regresión
        ruta_clasificacion : str
            Ruta para guardar el modelo de clasificación
        """
        import pickle
        
        if self.modelo_regresion is not None:
            with open(ruta_regresion, 'wb') as f:
                pickle.dump({
                    'modelo': self.modelo_regresion,
                    'scaler': self.scaler,
                    'caracteristicas': self.X_nombres
                }, f)
            print(f"Modelo de regresión guardado en: {ruta_regresion}")
        
        if self.modelo_clasificacion is not None:
            with open(ruta_clasificacion, 'wb') as f:
                pickle.dump({
                    'modelo': self.modelo_clasificacion,
                    'scaler': self.scaler,
                    'caracteristicas': self.X_nombres,
                    'umbral': self.umbral_clasificacion
                }, f)
            print(f"Modelo de clasificación guardado en: {ruta_clasificacion}")
    
    @classmethod
    def cargar_modelo(cls, ruta, tipo='regresion'):
        """
        Carga un modelo guardado
        
        Parámetros:
        -----------
        ruta : str
            Ruta del archivo del modelo
        tipo : str
            Tipo de modelo ('regresion' o 'clasificacion')
        
        Retorna:
        --------
        ModeloHumedad : Instancia con el modelo cargado
        """
        import pickle
        
        # Crear instancia vacía
        modelo = cls(pd.DataFrame({'HumedadFinal_pct': []}))
        
        # Cargar modelo guardado
        with open(ruta, 'rb') as f:
            datos = pickle.load(f)
        
        # Restaurar atributos
        modelo.scaler = datos['scaler']
        modelo.X_nombres = datos['caracteristicas']
        
        if tipo == 'regresion':
            modelo.modelo_regresion = datos['modelo']
        else:
            modelo.modelo_clasificacion = datos['modelo']
            modelo.umbral_clasificacion = datos.get('umbral', 5.0)
        
        return modelo


 