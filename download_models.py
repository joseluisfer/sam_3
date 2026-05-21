import os
import sys

print("-> Iniciando script de precarga de modelos...", flush=True)

try:
    print("-> Intentando importar clases de Transformers...", flush=True)
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor
except ImportError as ie:
    print(f"\n❌ ERROR DE IMPORTACIÓN: {str(ie)}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    repo = 'facebook/sam3'
    print(f"-> Descargando pesos y scripts desde: '{repo}'...", flush=True)
    
    # Añadimos trust_remote_code=True por seguridad de la arquitectura externa
    print("   ... Descargando Sam3Model", flush=True)
    Sam3Model.from_pretrained(repo, trust_remote_code=True)
    
    print("   ... Descargando Sam3Processor", flush=True)
    Sam3Processor.from_pretrained(repo, trust_remote_code=True)
    
    print("   ... Descargando Sam3TrackerModel", flush=True)
    Sam3TrackerModel.from_pretrained(repo, trust_remote_code=True)
    
    print("   ... Descargando Sam3TrackerProcessor", flush=True)
    Sam3TrackerProcessor.from_pretrained(repo, trust_remote_code=True)
    
    print("✅ ¡Todo se ha precargado correctamente en /app/cache!", flush=True)

except Exception as e:
    print(f"\n❌ ERROR EN DESCARGA: {str(e)}", file=sys.stderr, flush=True)
    sys.exit(1)
