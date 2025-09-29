#!/bin/bash

echo "Starting retraining of DreamWaltz-G for Clown"
bash scripts/train_w_expr.sh "creepy clown"
#bash scripts/train_w_expr.sh "creepy clown"

echo "Starting retraining of DreamWaltz-G for Ghost"
bash scripts/train_w_expr.sh "victorian era ghost"
#bash scripts/train_w_expr.sh "victorian era ghost"


echo "Starting retraining of DreamWaltz-G for Rapunzel"
bash scripts/train_wo_expr.sh "Rapunzel in Tangled"
#bash scripts/train_wo_expr.sh "Rapunzel in Tangled"

echo "Starting retraining of DreamWaltz-G for Goku"
bash scripts/train_wo_expr.sh "Goku"
#bash scripts/train_wo_expr.sh "Goku"

echo "Starting retraining of DreamWaltz-G for Alucard (complex clothing)"
bash scripts/train_wo_expr.sh "Alucard in Hellsing"
#bash scripts/train_w_expr.sh "Alucard in Hellsing"

echo "Starting retraining of DreamWaltz-G for Clown"
bash scripts/train_wo_expr.sh "creepy clown"
#bash scripts/train_wo_expr.sh "creepy clown"

echo "Starting retraining of DreamWaltz-G for Guts (complex body structure)"
bash scripts/train_wo_expr.sh "Guts in Berserk"
#bash scripts/train_wo_expr.sh "Guts in Berserk"

echo "Starting retraining of DreamWaltz-G for Shoto Todoroki (complex body structure and details)"
bash scripts/train_wo_expr.sh "Shoto Todoroki in My Hero Academia"
#bash scripts/train_wo_expr.sh "Shoto Todoroki in My Hero Academia"


echo "Starting retraining of DreamWaltz-G for Belisarius Cawl (very complex body structure)"
bash scripts/train_wo_expr.sh "Belisarius Cawl in Warhammer"
#bash scripts/train_wo_expr.sh "Belisarius Cawl in Warhammer"

echo "Starting retraining of DreamWaltz-G for Space Marine (complex body structure)"
bash scripts/train_wo_expr.sh "Space Marine in Warhammer"
#bash scripts/train_wo_expr.sh "Space Marine in Warhammer"

echo "Starting retraining of DreamWaltz-G for Pipimi (non-human proportions)"
bash scripts/train_wo_expr.sh "Pipimi in Pop Team Epic"
#bash scripts/train_wo_expr.sh "Pipimi in Pop Team Epic"

echo "Starting retraining of DreamWaltz-G for Chihiro Ogino (distinctive clothing and features)"
bash scripts/train_wo_expr.sh "Chihiro Ogino in Spirited Away"
#bash scripts/train_wo_expr.sh "Chihiro Ogino in Spirited Away"

echo "Starting retraining of DreamWaltz-G for different designs and hair animation"
bash scripts/train_w_expr.sh "an anime girl with vibrant pink hair styled in long, flowing curls and wearing a gothic lolita dress"
#bash scripts/train_w_expr.sh "an anime girl with vibrant pink hair styled in long, flowing curls and wearing a gothic lolita dress"

echo "Starting retraining of DreamWaltz-G for costumes"
bash scripts/train_w_expr.sh "a humanoid cactus wearing a sombrero"
#bash scripts/train_w_expr.sh "a humanoid cactus wearing a sombrero"

echo "Starting retraining of DreamWaltz-G for contrasting designs"
bash scripts/train_w_expr.sh "a man with a split personality, each side of the body dressed in completely different styles"
#bash scripts/train_w_expr.sh "a man with a split personality, each side of the body dressed in completely different styles"

echo "Starting retraining of DreamWaltz-G for complex clothing"
bash scripts/train_w_expr.sh "a ninja with a scarf that seems to have a life of its own"
#bash scripts/train_w_expr.sh "a ninja with a scarf that seems to have a life of its own"

echo "Starting retraining of DreamWaltz-G for futuristic cyborg aesthetics"
bash scripts/train_wo_expr.sh "a futuristic cyborg with sleek metallic armor and glowing blue eyes"
#bash scripts/train_wo_expr.sh "a futuristic cyborg with sleek metallic armor and glowing blue eyes"

echo "Starting retraining of DreamWaltz-G for historical figures"
bash scripts/train_wo_expr.sh "a medieval knight with a polished silver breastplate and a red cape"
#bash scripts/train_wo_expr.sh "a medieval knight with a polished silver breastplate and a red cape"

echo "Starting retraining of DreamWaltz-G for robotic pets"
bash scripts/train_wo_expr.sh "a small robotic dog with neon accents and a metallic body"
#bash scripts/train_wo_expr.sh "a small robotic dog with neon accents and a metallic body"

echo "Starting retraining of DreamWaltz-G for whimsical animal characters"
bash scripts/train_wo_expr.sh "a playful cat wearing a magician's hat and cape, holding a wand"
#bash scripts/train_w_expr.sh "a playful cat wearing a magician's hat and cape, holding a wand"
