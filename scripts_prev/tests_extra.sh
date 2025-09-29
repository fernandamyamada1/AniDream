#!/bin/bash

echo "Starting retraining of DreamWaltz-G for anime and low poly"
bash scripts/train_w_expr.sh "an anime boy in low poly style"

echo "Starting retraining of DreamWaltz-G for different designs and hair animation"
bash scripts/train_w_expr.sh "an anime girl with vibrant pink hair styled in long, flowing curls and wearing a gothic lolita dress"

echo "Starting retraining of DreamWaltz-G for contrasting designs"
bash scripts/train_w_expr.sh "a man with a split personality, each side of the body dressed in completely different styles"

echo "Starting retraining of DreamWaltz-G for low poly"
bash scripts/train_wo_expr.sh "a man in low poly style"

echo "Starting retraining of DreamWaltz-G for Robotic Dog"
bash scripts/train_wo_expr.sh "a small robotic dog with neon accents and a metalic body"

echo "Starting retraining of DreamWaltz-G for Rapunzel"
bash scripts/train_wo_expr.sh "Rapunzel in Tangled"

echo "Starting retraining of DreamWaltz-G for Goku"
bash scripts/train_wo_expr.sh "Goku"

echo "Starting retraining of DreamWaltz-G for Shoto Todoroki (complex body structure and details)"
bash scripts/train_wo_expr.sh "Shoto Todoroki in My Hero Academia"

echo "Starting retraining of DreamWaltz-G for Belisarius Cawl (very complex body structure)"
bash scripts/train_wo_expr.sh "Belisarius Cawl in Warhammer"

echo "Starting retraining of DreamWaltz-G for Alucard (complex clothing)"
bash scripts/train_w_expr.sh "Alucard in Hellsing"

echo "Starting retraining of DreamWaltz-G for Sailor Moon (complex hairstyle)"
bash scripts/train_wo_expr.sh "Sailor Moon"

echo "Starting retraining of DreamWaltz-G for Guts (complex body structure)"
bash scripts/train_wo_expr.sh "Guts in Berserk"







# Add more commands as needed
