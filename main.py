# ===== main.py unifié =====

import argparse
import os
from mobile_sam_podiatry import MobileSAMPodiatryPipeline, quick_measure, batch_process_folder, validate_setup

def main():
    """Interface ligne de commande pour MobileSAMPodiatryPipeline"""
    parser = argparse.ArgumentParser(
        description="📏 SAM Podiatry - Scan pied + carte pour mesures précises",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --top top.jpg                      # Vue dessus: largeur + toe_angle
  python main.py --side-only side.jpg --side right  # Vue profil: longueur heel-toe
  python main.py --hybrid top.jpg side.jpg          # Mesure combinée
  python main.py photo.jpg                          # Mesure générique (legacy)
  python main.py --validate                         # Vérifier installation
"""
    )

    parser.add_argument('image', nargs='?', help="Image à analyser")
    parser.add_argument('--debug', action='store_true', help="Sauver images debug")
    parser.add_argument('--batch', metavar='FOLDER', help="Traiter un dossier")
    parser.add_argument('--output', metavar='CSV', help="Fichier CSV pour --batch")
    parser.add_argument('--validate', action='store_true', help="Vérifier installation")
    parser.add_argument('--hybrid', nargs=2, metavar=('TOP', 'SIDE'),
                        help="Mesure combinée vue dessus + profil")
    parser.add_argument('--top', metavar='IMAGE',
                        help="Vue dessus uniquement (largeur + toe_angle) - ARUCO ONLY")
    parser.add_argument('--side-only', metavar='IMAGE', dest='side_only',
                        help="Vue profil uniquement (longueur heel-toe) - ARUCO ONLY")
    parser.add_argument('--side', choices=['left', 'right'], default='right',
                        help="Côté du pied (left/right). Défaut: right")
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

    # Cas : Mesure hybride top+side
    if args.hybrid:
        top_img, side_img = args.hybrid
        if not os.path.exists(top_img) or not os.path.exists(side_img):
            print("❌ Fichiers top/side manquants")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        print(f"🦶 Analyse combinée (Hybrid)...")
        
        # 1. Traitement TOP
        print(f"\n--- [1/2] Traitement VUE DESSUS ({os.path.basename(top_img)}) ---")
        top_result = pipeline.process_top_view(top_img, debug=args.debug)
        
        # 2. Traitement SIDE
        print(f"\n--- [2/2] Traitement VUE PROFIL ({os.path.basename(side_img)}) ---")
        side_result = pipeline.process_side_view(side_img, debug=args.debug, foot_side=args.side)

        if 'error' in top_result:
            print(f"❌ Erreur TOP: {top_result['error']}")
            return
        if 'error' in side_result:
            print(f"❌ Erreur SIDE: {side_result['error']}")
            return

        # Fusion des résultats
        result = {
            'foot_side': args.side,
            'length_cm': side_result['length_cm'],
            'width_cm': top_result['width_cm'],
            'toe_angle_deg': top_result.get('toe_angle_deg', 0),
            'top_file': top_img,
            'side_file': side_img
        }

        print("\n✅ MESURES COMBINÉES :")
        print(f"🦶 Côté : {result['foot_side']}")
        print(f"📏 Longueur: {result['length_cm']} cm")
        print(f"📐 Largeur : {result['width_cm']} cm")
        # print(f"🦶 Angle orteils: {result['toe_angle_deg']}°")
        
        # Output JSON format for mobile integration
        import json
        print(f"\n📱 JSON pour intégration mobile:")
        print(json.dumps(result, indent=2))

        if args.debug:
            print("📁 Images debug sauvegardées dans le dossier output/")
        return

    # Cas : Vue profil uniquement (side)
    if args.side_only:
        if not os.path.exists(args.side_only):
            print(f"❌ Fichier introuvable: {args.side_only}")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        print(f"👁️ Analyse vue PROFIL (side view)...")
        result = pipeline.process_side_view(args.side_only, debug=args.debug)

        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
        else:
            print("\n✅ MESURES VUE PROFIL :")
            print(f"📏 Longueur : {result['length_cm']} cm")

            if args.debug:
                print("📁 Images debug sauvegardées dans le dossier output/")
        return

    # Cas : Vue dessus uniquement (top)
    if args.top:
        if not os.path.exists(args.top):
            print(f"❌ Fichier introuvable: {args.top}")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        print("👁️ Analyse vue DESSUS (top view)...")
        result = pipeline.process_top_view(args.top, debug=args.debug, foot_side=args.side)

        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
        else:
            print("\n✅ MESURES VUE DESSUS :")
            print(f"📐 Largeur : {result['width_cm']} cm")
            # print(f"🦶 Angle orteils: {result['toe_angle_deg']}°") # Optional now

            if args.debug:
                print("📁 Images debug sauvegardées dans le dossier output/")
        return

    # Cas : Vue profil uniquement (side)
    if args.side_only:
        if not os.path.exists(args.side_only):
            print(f"❌ Fichier introuvable: {args.side_only}")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        print(f"👁️ Analyse vue PROFIL (side view)...")
        result = pipeline.process_side_view(args.side_only, debug=args.debug, foot_side=args.side)

        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
        else:
            print("\n✅ MESURES VUE PROFIL :")
            print(f"📏 Longueur : {result['length_cm']} cm")

            if args.debug:
                print("📁 Images debug sauvegardées dans le dossier output/")
        return

    # Cas : Image unique (Défaut -> Side View)
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Fichier introuvable: {args.image}")
            return

        print(f"🚀 Initialisation SAM ({args.model}) ...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)

        if not pipeline.initialized:
            print("❌ SAM non initialisé. Vérifiez `pip install segment-anything` et le modèle.")
            return

        print(f"ℹ️ Mode par défaut : Analyse VUE PROFIL (Side View - Pied {args.side})")
        result = pipeline.process_side_view(args.image, debug=args.debug, foot_side=args.side)

        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
        else:
            print("\n✅ MESURES :")
            print(f"📏 Longueur: {result['length_cm']} cm")
            print(f"✨ Confiance : {result['confidence']}%")

            if args.debug:
                print("📁 Images debug sauvegardées dans le dossier output/")

    else:
        parser.print_help()
        print("\n💡 Astuce : Essayez `python main.py image.jpg` (Profil) ou `--top image.jpg` (Dessus)")

if __name__ == "__main__":
    main()