# ===== main.py unifié =====

import argparse
import os
from pydantic import BaseModel, ValidationError, validator
from mobile_sam_podiatry import MobileSAMPodiatryPipeline, quick_measure, batch_process_folder, validate_setup


class FaceImages(BaseModel):
    top: str
    left: str
    right: str
    front: str
    back: str

    @validator('*')
    def validate_image(cls, v):
        if not os.path.exists(v):
            raise ValueError('fichier introuvable')
        if not v.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise ValueError('format image non supporté')
        return v


def prompt_image(text):
    while True:
        path = input(text)
        if os.path.exists(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return path
        print('❌ chemin invalide, réessayez...')

def main():
    """Interface ligne de commande pour MobileSAMPodiatryPipeline"""
    parser = argparse.ArgumentParser(
        description="📏 SAM Podiatry - Scan pied + carte pour mesures précises",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py photo.jpg             # Mesure une image
  python main.py photo.jpg --debug     # Sauvegarder debug
  python main.py --batch dossier/      # Traiter dossier complet
  python main.py --validate            # Vérifier installation
  python main.py --top dessus.jpg --left gauche.jpg --right droit.jpg --front avant.jpg --back arriere.jpg
"""
    )

    parser.add_argument('image', nargs='?', help="Image à analyser")
    parser.add_argument('--pair', nargs=2, metavar=('GAUCHE','DROIT'),
                        help="Mesurer pied gauche et droit")
    parser.add_argument('--debug', action='store_true', help="Sauver images debug")
    parser.add_argument('--batch', metavar='FOLDER', help="Traiter un dossier")
    parser.add_argument('--output', metavar='CSV', help="Fichier CSV pour --batch")
    parser.add_argument('--validate', action='store_true', help="Vérifier installation")
    parser.add_argument('--model', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b',
                        help="Modèle SAM (vit_b par défaut)")
    parser.add_argument('--top', help="Image vue du dessus")
    parser.add_argument('--left', help="Image côté gauche")
    parser.add_argument('--right', help="Image côté droit")
    parser.add_argument('--front', help="Image face avant")
    parser.add_argument('--back', help="Image face arrière")

    args = parser.parse_args()

    # Cas : Vérifier installation
    if args.validate:
        validate_setup()
        return

    # Cas : Batch
    if args.batch:
        batch_process_folder(args.batch, args.output)
        return

    # Cas : Paire gauche/droit
    if args.pair:
        left, right = args.pair
        if not os.path.exists(left) or not os.path.exists(right):
            print("❌ Fichiers gauche/droit introuvables")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        res_left = pipeline.process_foot_image(left, debug=args.debug)
        res_right = pipeline.process_foot_image(right, debug=args.debug)

        def show(title, res):
            print(f"\n=== {title} ===")
            if 'error' in res:
                print(f"❌ Erreur: {res['error']}")
                return
            print(f"📏 Longueur: {res['length_cm']} cm")
            print(f"📐 Largeur : {res['width_cm']} cm")
            print(f"🔢 Ratio L/l : {res['length_width_ratio']}")
            print(f"📊 Surface : {res['area_cm2']} cm²")
            print(f"🔄 Périmètre : {res['perimeter_cm']} cm")
            print(f"✨ Confiance : {res['confidence']}%")

        show('Pied gauche', res_left)
        show('Pied droit', res_right)
        return

    # Cas : Mesure de toutes les faces
    if args.top or args.left or args.right or args.front or args.back:
        images = {}
        prompts = {
            'top': 'Image vue du dessus : ',
            'left': 'Image côté gauche : ',
            'right': 'Image côté droit : ',
            'front': 'Image face avant : ',
            'back': 'Image face arrière : '
        }

        for ori in ['top', 'left', 'right', 'front', 'back']:
            img = getattr(args, ori)
            if not img:
                img = prompt_image(prompts[ori])
            images[ori] = img

        while True:
            try:
                faces = FaceImages(**images)
                break
            except ValidationError as e:
                print(e)
                for err in e.errors():
                    field = err['loc'][0]
                    images[field] = prompt_image(f"Nouvelle image pour {field} : ")

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        results = {}
        for orientation, path in faces.dict().items():
            if orientation == 'top':
                res = pipeline.process_foot_image(path, debug=args.debug)
            else:
                res = pipeline.process_face_image(path, orientation, debug=args.debug)
            results[orientation] = res

        for orientation, res in results.items():
            print(f"\n=== {orientation.upper()} ===")
            if 'error' in res:
                print(f"❌ Erreur: {res['error']}")
                continue
            if orientation == 'top':
                print(f"📏 Longueur: {res['length_cm']} cm")
                print(f"📐 Largeur : {res['width_cm']} cm")
            else:
                print(f"📏 Hauteur : {res['height_cm']} cm")
                print(f"📐 Largeur : {res['width_cm']} cm")
            print(f"✨ Confiance : {res['confidence']}%")
        return

    # Cas : Image unique
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Fichier introuvable: {args.image}")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        result = pipeline.process_foot_image(args.image, debug=True)

        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
        else:
            print("\n✅ MESURES :")
            print(f"📏 Longueur: {result['length_cm']} cm")
            print(f"📐 Largeur : {result['width_cm']} cm")
            print(f"🔢 Ratio L/l : {result['length_width_ratio']}")
            print(f"📊 Surface : {result['area_cm2']} cm²")
            print(f"🔄 Périmètre : {result['perimeter_cm']} cm")
            print(f"✨ Confiance : {result['confidence']}%")

            if args.debug:
                print("📁 Images debug sauvegardées dans le dossier output/")

    else:
        parser.print_help()
        print("\n💡 Astuce : Essayez `python main.py image.jpg` ou `--batch dossier/`")

if __name__ == "__main__":
    main()
