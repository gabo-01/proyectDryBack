"""
API para servir el modelo de predicción de humedad en filtros de secado.
Este script crea un endpoint que puede ser consumido por una aplicación móvil.
"""

from flask import Flask, request, jsonify
import traceback
import pickle
import os
import json
import pandas as pd  # Added missing import
 
from ML.ModelTraining.ModeloHumedad import ModeloHumedad
import smtplib
from email.mime.text import MIMEText
import requests
 
app = Flask(__name__)
MODELS_FOLDER = '/Users/fernando/Documents/Big data E IA/Semestre 1/Mineria de Datos/'
 
modelo_regresion = None
modelo_clasificacion = None
 

GEMINI_API_KEY = "AIzaSyAVl5sdhcLfbtGiwEcn-mS__wGy7qNVuTA"
GMAIL_USER = "gabrielmtzg76@gmail.com"
GMAIL_PASSWORD = "jhtx ysxf mwmo rgcz"
def load_models():
    """Load models from disk if available"""
    global modelo_regresion, modelo_clasificacion, predictor
    
    regresion_path = os.path.join(MODELS_FOLDER, 'modelo_humedad_regresion.pkl')
    clasificacion_path = os.path.join(MODELS_FOLDER, 'modelo_humedad_clasificacion.pkl')
   
    
    try:
        
        if os.path.exists(regresion_path):
            modelo_regresion = ModeloHumedad.cargar_modelo(regresion_path, tipo='regresion')
            print("Modelo de regresión cargado con éxito")
        else:
            print(f"ADVERTENCIA: El modelo de regresión no existe en la ruta {regresion_path}")
        
        # Cargar modelo de clasificación
        if os.path.exists(clasificacion_path):
            modelo_clasificacion = ModeloHumedad.cargar_modelo(clasificacion_path, tipo='clasificacion')
            print("Modelo de clasificación cargado con éxito")
        else:
            print(f"ADVERTENCIA: El modelo de clasificación no existe en la ruta {clasificacion_path}")
    
    except Exception as e:
        print(f"Error al cargar modelos: {str(e)}")
        traceback.print_exc()

# Llamar a la función para cargar los modelos al iniciar la aplicación
load_models()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar el estado de la API y los modelos cargados
    """
    models_status = {
        
        "modelo_regresion": modelo_regresion is not None,
        "modelo_clasificacion": modelo_clasificacion is not None
    }
    
    return jsonify({
        "status": "OK",
        "models_loaded": models_status
    })

@app.route('/predict_complete', methods=['POST'])
def predict_complete():
    """
    Endpoint combinado que devuelve regresión, clasificación, probabilidades
    y decisión sobre si se necesita otro ciclo
    """
    global modelo_regresion, modelo_clasificacion
    
    # Verificar que ambos modelos estén disponibles
    if modelo_regresion is None:
        return jsonify({
            'error': 'Modelo de regresión no disponible. Por favor, entrene un modelo primero.'
        }), 400
        
    if modelo_clasificacion is None:
        return jsonify({
            'error': 'Modelo de clasificación no disponible. Por favor, entrene un modelo primero.'
        }), 400
    
    try:
        # Obtener datos de la solicitud
        data = request.json
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos para la predicción'}), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([data])
        
        # Verificar características para regresión
        reg_required_features = set(modelo_regresion.X_nombres)
        provided_features = set(df.columns)
        
        reg_missing_features = reg_required_features - provided_features
        if reg_missing_features:
            return jsonify({
                'error': f'Faltan características requeridas para regresión: {", ".join(reg_missing_features)}',
                'required_features': list(reg_required_features)
            }), 400
            
        # Verificar características para clasificación
        clas_required_features = set(modelo_clasificacion.X_nombres)
        clas_missing_features = clas_required_features - provided_features
        if clas_missing_features:
            return jsonify({
                'error': f'Faltan características requeridas para clasificación: {", ".join(clas_missing_features)}',
                'required_features': list(clas_required_features)
            }), 400
        
        # Hacer predicciones
        regression_prediction = modelo_regresion.predecir(df, tipo='regresion')
        classification_prediction = modelo_clasificacion.predecir(df, tipo='clasificacion')
        probability_prediction = modelo_clasificacion.predecir_probabilidad(df)
        
        # Extraer valores
        humidity_value = float(regression_prediction[0])
        class_value = int(classification_prediction[0])
        class_label = "Alta" if class_value == 1 else "Baja"
        prob_low = float(probability_prediction[0][0])
        prob_high = float(probability_prediction[0][1])
        
        # Umbral de confianza para determinar si se necesita otro ciclo
        # Este umbral puede ajustarse según tus necesidades
        confidence_threshold = 0.7
        max_confidence = max(prob_low, prob_high)
        needs_another_cycle = max_confidence < confidence_threshold
        
        # También puedes agregar lógica para decidir basada en el valor de humedad predicho
        humidity_threshold = 0.5  # Umbral ejemplo, ajustar según necesidades
        if humidity_value > humidity_threshold and class_value == 1:  # Alta humedad
            needs_another_cycle = True
            
        return jsonify({
            'prediction': humidity_value,
            'prediction_class': class_value,
            'prediction_label': class_label,
            'probability_low': prob_low,
            'probability_high': prob_high,
            'confidence': max_confidence,
            'needs_another_cycle': needs_another_cycle,
            'message': f'Humedad predicha: {humidity_value:.4f}%, Clase: {class_label}, ' +
                      f'Confianza: {max_confidence:.2f}, ' +
                      f'{"Necesita" if needs_another_cycle else "No necesita"} otro ciclo'
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error en predict_complete: {str(e)}")
        print(error_details)
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500



def redactar_mensaje_con_gemini(payload):
    import json

    prompt = (
        f"Tengo una predicción de humedad final de {payload['humedad_predicha']:.2f}% "
        f"y la clase predicha es {payload['clase_predicha']}, lo que significa que "
        f"{'NO' if payload['clase_predicha'] == 0 else 'SÍ'} se necesita otro ciclo de secado. "
        f"La confianza es del {payload['confianza']*100:.1f}%. Redacta un correo profesional informando este resultado."
    )

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    return result["candidates"][0]["content"]["parts"][0]["text"]


# Envía correo por Gmail
def enviar_correo(destinatario, asunto, cuerpo):
    msg = MIMEText(cuerpo, "plain", "utf-8")
    msg["Subject"] = asunto
    msg["From"] = GMAIL_USER
    msg["To"] = destinatario

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.send_message(msg)

# Endpoint principal
@app.route("/api/send_gemini_email", methods=["POST"])
 
def send_gemini_email():
    try:
        payload = request.get_json()
        print("Payload recibido:", payload)

        cuerpo = redactar_mensaje_con_gemini(payload)

        enviar_correo(
            destinatario="gabrielmtzg76@gmail.com",
            asunto="Resultado del ciclo de secado",
            cuerpo=cuerpo
        )

        return jsonify({"status": "ok", "message": "Correo enviado exitosamente"})
    except Exception as e:
        import traceback
        traceback.print_exc()  # Muestra la traza completa en consola
        return jsonify({"status": "error", "message": str(e)}), 500
0


if __name__ == '__main__':
    # Iniciar la aplicación en modo desarrollo
    # En producción, usa un servidor WSGI como Gunicorn
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)