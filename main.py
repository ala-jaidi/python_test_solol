# ===== main.py unifié =====

import argparse
import os
from dataclasses import dataclass
from mobile_sam_podiatry import (
    MobileSAMPodiatryPipeline,
    quick_measure,
    batch_process_folder,
    validate_setup,
)


@dataclass
class FaceImages:
    """Container for foot photos used in validation tests."""

    top: str
    left: str
    right: str
    front: str
    back: str

    def __post_init__(self) -> None:
        for field in ("top", "left", "right", "front", "back"):
            path = getattr(self, field)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

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
"""
    )

    parser.add_argument('image', nargs='?', help="Image à analyser")
    parser.add_argument('--debug', action='store_true', help="Sauver images debug")
    parser.add_argument('--batch', metavar='FOLDER', help="Traiter un dossier")
    parser.add_argument('--output', metavar='CSV', help="Fichier CSV pour --batch")
    parser.add_argument('--validate', action='store_true', help="Vérifier installation")
    parser.add_argument('--model', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b',
                        help="Modèle SAM (vit_b par défaut)")

    args = parser.parse_args()

    # Cas : Vérifier installation
    if args.validate:
        validate_setup()
        return

    # Cas : Batch
    if args.batch:
        batch_process_folder(args.batch, args.output)
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
