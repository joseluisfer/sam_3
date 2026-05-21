import os
import sys

print("-> Iniciando script de precarga de modelos...", flush=True)

try:
    # 1. Comprobar si las clases existen en la librería instalada
    print("-> Comprobando disponibilidad de clases SAM 3 en transformers...", flush=True)
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor
except ImportError as ie:
    print(f"\n❌ ERROR DE IMPORTACIÓN: Tu versión de 'transformers' no reconoce los modelos SAM 3 nativos.", file=sys.stderr, flush=True)
    print(f"Detalle: {str(ie)}", file=sys.stderr, flush=True)
    print("Sugerencia: Revisa si el Space de HF usa otra librería o requiere instalar transformers desde una rama específica de git.", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    repo = 'facebook/sam3'
    print(f"-> Descargando pesos desde el repositorio: '{repo}'...", flush=True)
    
    print("   ... Descargando Sam3Model", flush=True)
    Sam3Model.from_pretrained(repo)
    
    print("   ... Descargando Sam3Processor", flush=True)
    Sam3Processor.from_pretrained(repo)
    
    print("   ... Descargando Sam3TrackerModel", flush=True)
    Sam3TrackerModel.from_pretrained(repo)
    
    print("   ... Descargando Sam3TrackerProcessor", flush=True)
    Sam3TrackerProcessor.from_pretrained(repo)
    
    print("✅ ¡Todo se ha precargado correctamente en /app/cache!", flush=True)

except Exception as e:
    print(f"\n❌ ERROR DURANTE LA DESCARGA DEL REPO '{repo}': {str(e)}", file=sys.stderr, flush=True)
    sys.exit(1)
