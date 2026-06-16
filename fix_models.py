import os
import zipfile
import json
import tempfile
import shutil

MODELS_DIR = r""
models = ["emodb_model.keras", "emovo_model.keras", "shemo_model.keras"]

def fix_json(obj):
    if isinstance(obj, dict):
        if "quantization_config" in obj:
            print(f"  Removing quantization_config from {obj.get('name', 'unnamed')}")
            del obj["quantization_config"]
        for k, v in obj.items():
            fix_json(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_json(item)

for model_name in models:
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        print(f"Skipping {model_name}: not found")
        continue
    
    print(f"Fixing {model_name}...")
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        config_path = os.path.join(temp_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            fix_json(config)
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Backup original if not already backed up
            backup_path = path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(path, backup_path)
            
            # Re-zip
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, temp_dir)
                        zip_ref.write(file_path, rel_path)
            print(f"  Successfully fixed {model_name}")
    except Exception as e:
        print(f"  FAILED to fix {model_name}: {e}")
    finally:
        shutil.rmtree(temp_dir)
