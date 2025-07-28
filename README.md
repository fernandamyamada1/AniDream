# AniDream: Generating Skeleton-Guided Anime Avatars from Text Prompts

![AniDream Output](assets/cover.png)

 ## News

**AniDream has been accepted for presentation at [IEEE ISMAR 2025](https://ieeeismar.org/)!** 

Stay tuned for the final paper and presentation details.

## Abstract

Generating high-quality anime avatars has become an increasingly important task in the fields of animation, gaming, and virtual reality. However, existing frameworks often face challenges in achieving anatomical consistency and mitigating visual artifacts. To address these limitations, we introduce AniDream, a novel framework designed for the generation of high-quality anime avatars. Unlike previous approaches that primarily relied on image-based inputs, AniDream incorporates text-guided generation, allowing users to create diverse anime avatars directly from text prompts. AniDream uses a skeleton-guided approach, ensuring anatomical consistency while focusing on refining attention around key regions. Our framework introduces a novel loss function that simulates a cel-shading effect and encourages the generated avatars to maintain sharp contour definitions and shadowing consistent with anime aesthetics. Experiments show that AniDream outperforms other frameworks by reducing artifacts and maintaining visual consistency across various poses and viewpoints. It also achieves an average CLIPScore of 33.07, demonstrating its effectiveness in closely aligning generated avatars with text prompts.

**keywords: anime, avatar, diffusion model, generative model, text-to-image, low-rank adaptation, cel-shading.**

## Getting Started

### Core Requirements

- Python 3.11.10
- PyTorch 2.1.0  
- CUDA 11.8  

### Installation

```bash
git clone https://github.com/fernandamyamada1/AniDream.git
cd AniDream
pip install -r requirements.txt

### Usage
bash scripts/anime_ismar.sh "an anime boy" 
