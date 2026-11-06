# eia-espia-mlops-taller3-fastapi

Repositorio oficial para el despliegue del pipeline de Machine Learning mediante FastAPI y systemd en AWS EC2.

## Instrucciones de Ejecución
1. Activar entorno conda:
   ```bash
   conda activate eia-espia-mlops-taller3-fastapi-env
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar servidor de desarrollo:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
4. Probar Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

