# SafeVision AR - Alertes de Sécurité en Temps Réel avec RA

**SafeVision AR** est une solution complète de surveillance de sécurité en temps réel utilisant la vision par ordinateur et la réalité augmentée. Le système capture les flux vidéo de plusieurs caméras, détecte les objets dangereux ou abandonnés et les comportements à risque, puis diffuse des alertes vers une application Flutter en AR pour une visualisation intuitive.

## Fonctionnalités

* **Support multi-caméras** : traitement parallèle de deux flux vidéo (ou plus)
* **Détection et suivi d’objets** : modèles YOLO pour les cigarettes, couteaux, valises, sacs à dos
* **Analyse de comportements** : détection de chute et alertes de tabagisme basées sur l’analyse de la pose et du maillage des lèvres
* **Réalité augmentée** : client Flutter AR affichant les alertes superposées dans le monde réel
* **Streaming WebSocket** : envoi d’alertes JSON et de trames encodées en base64 à faible latence

## Structure du projet

```
├── codes/                 # Scripts Python pour la détection et le streaming
│   ├── detect_live_pose_roi_lips_dual.py
│   └── ...
├── models/                # Modèles YOLO pré-entraînés (.pt et .onnx)
│   ├── best (10).pt
│   ├── yolo11n.onnx
│   └── yolo11n-pose.onnx
├── safevision_ar/         # Application Flutter AR
│   ├── lib/
│   ├── assets/
│   └── pubspec.yaml
├── .gitattributes         # Attributs Git
└── README.md              # Documentation du projet
```

## Prérequis

* **Python 3.9+** avec :

  * OpenCV
  * NumPy
  * MediaPipe
  * Ultralytics YOLO
  * ONNX Runtime (ou onnxruntime-gpu)
* **Flutter SDK** 3.x
* Appareil Android/iOS compatible ARCore/ARKit

## Installation

### 1. Environnement Python

```bash
cd codes
pip install -r requirements.txt
```

### 2. Application Flutter

```bash
cd safevision_ar
flutter pub get
```

## Utilisation

### 1. Lancer le serveur CV Python

```bash
python detect_live_pose_roi_lips_dual.py
```

* Ouvre deux flux caméra (indices 1 et 3 par défaut)
* Exécute la détection, l’annotation et le suivi
* Stream d’alertes JSON et de trames base64 via WebSocket sur le port 8765

### 2. Démarrer le client Flutter AR

```bash
cd safevision_ar
flutter run
```

* Se connecte à `ws://<adresse-serveur>:8765`
* Reçoit les flux et alertes
* Affiche des overlays AR (icônes, messages) ancrés sur les objets/personnes détectés

## Configuration

* Modifiez les **indices caméra** et la **résolution d’affichage** dans `detect_live_pose_roi_lips_dual.py`
* Ajustez les **seuils de confiance** dans le dictionnaire `CONF_THRESH`
* Définissez l’**URL WebSocket** dans `safevision_ar/lib/main.dart`

## Contribuer

1. Forkez le dépôt
2. Créez une branche de fonctionnalité (`git checkout -b feature/VotreFonctionnalité`)
3. Commitez vos modifications (`git commit -m "Ajout de ma fonctionnalité"`)
4. Pushez sur votre branche (`git push origin feature/VotreFonctionnalité`)
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Développé par Rostom Ben-Abdallah & contributeurs*
