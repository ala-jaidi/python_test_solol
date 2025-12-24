
import requests
import json
import os
import sys

# Configuration
API_URL = "http://127.0.0.1:8000/measure"
TOP_VIEW_PATH = "data/ad.jpg"   # Vue de dessus (Carte crédit)
SIDE_VIEW_PATH = "data/ar2.jpeg" # Vue de profil (ArUco)
OUTPUT_DIR = "client_downloads"

def test_api():
    if not os.path.exists(TOP_VIEW_PATH) or not os.path.exists(SIDE_VIEW_PATH):
        print(f"❌ Images de test manquantes: {TOP_VIEW_PATH} ou {SIDE_VIEW_PATH}")
        return

    print(f"🚀 Envoi des images vers {API_URL}...")
    print(f"   - Top: {TOP_VIEW_PATH}")
    print(f"   - Side: {SIDE_VIEW_PATH}")

    # Prepare multipart upload
    files = {
        'top_view': open(TOP_VIEW_PATH, 'rb'),
        'side_view': open(SIDE_VIEW_PATH, 'rb')
    }
    data = {
        'foot_side': 'right'
    }

    try:
        response = requests.post(API_URL, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Succès ! Réponse du serveur:")
            print(json.dumps(result, indent=2))
            
            # Download files
            if 'data' in result and 'files' in result['data']:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                file_links = result['data']['files']
                
                print("\n📥 Téléchargement des fichiers DXF...")
                for key, url in file_links.items():
                    if url:
                        # Construct full URL if relative
                        if url.startswith("/"):
                            download_url = f"http://127.0.0.1:8000{url}"
                        else:
                            download_url = url
                            
                        filename = os.path.basename(url)
                        save_path = os.path.join(OUTPUT_DIR, filename)
                        
                        print(f"   - Téléchargement {key} -> {save_path}")
                        download_file(download_url, save_path)
            
        else:
            print(f"❌ Erreur {response.status_code}:")
            print(response.text)

    except Exception as e:
        print(f"❌ Exception lors de la requête: {e}")
    finally:
        files['top_view'].close()
        files['side_view'].close()

def download_file(url, save_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"     ✅ Sauvegardé")
    except Exception as e:
        print(f"     ❌ Erreur téléchargement: {e}")

if __name__ == "__main__":
    test_api()
