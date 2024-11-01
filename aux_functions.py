import json
from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def simulate_model_prediction(input_json):
    """
    Simula la llamada a un modelo de ML ya entrenado para devolver un resultado de clasificación.
    
    Args:
        input_json (str): Cadena JSON con 10 parámetros que serán usados para la predicción.
    
    Returns:
        dict: Resultado simulado de la clasificación.
    """
    
    # Convertir el string JSON a un diccionario de Python
    input_data = json.loads(input_json)

    # Verificar que el JSON tiene exactamente 10 parámetros
    if len(input_data) != 10:
        raise ValueError("El JSON de entrada debe contener exactamente 10 parámetros.")
    
    # Simular el proceso de predicción
    # Aquí podrías realizar cualquier procesamiento necesario
    # Por ahora, devolveremos una clasificación simulada
    simulated_prediction = {
        "predicción": "Clase A",  # Simula una predicción de clase
        "probabilidad": 0.85      # Simula una probabilidad asociada a la predicción
    }
    
    # Devolver el resultado de la simulación
    return simulated_prediction

def get_db_from_uri(uri):
    db = SQLDatabase.from_uri(f"sqlite:///db/{uri}")
    return db

class DB_Connection:

    def __init__(self):
        self.db = None
        self.db_name = None

    def get_db(self):
        db = SQLDatabase.from_uri(f"sqlite:///db/{self.db_name}")
        return db
    
    def upload_db_from_settings(self, file, table_name, sep, encoding):
        conn = sqlite3.connect(fr'db/{table_name}.db')
        if "csv" in file.name:
            df = pd.read_csv(file, sep=sep, encoding=encoding)
        else:
            df = pd.read_excel(file, sheet_name=sep)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        # Cierra la conexión
        conn.close()
        return df

db_connection = DB_Connection()

