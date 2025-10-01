# AniDream

🌟 Welcome to the AniDream Repository! 🌟 

We are excited to share the official repository for **"AniDream: Generating Skeleton-Guided Anime Avatars from Text Prompts"**



![AniDream Output](assets/cover.png)

🚀 Exciting News! 🚀

**AniDream has been accepted for presentation at [IEEE ISMAR 2025](https://ieeeismar.org/)!** 

💡 Title: AniDream: Generating Skeleton-Guided Anime Avatars from Text Prompts

👩‍💻 Authors: Fernanda Miyuki Yamada, Hiroki Takahashi

🎓 Conference: IEEE International Symposium on Mixed and Augmented Reality (ISMAR 2025)



## Abstract

Generating high-quality anime avatars has become an increasingly important task in the fields of animation, gaming, and virtual reality. However, existing frameworks often face challenges in achieving anatomical consistency and mitigating visual artifacts. To address these limitations, we introduce AniDream, a novel framework designed for the generation of high-quality anime avatars. Unlike previous approaches that primarily relied on image-based inputs, AniDream incorporates text-guided generation, allowing users to create diverse anime avatars directly from text prompts. AniDream uses a skeleton-guided approach, ensuring anatomical consistency while focusing on refining attention around key regions. Our framework introduces a novel loss function that simulates a cel-shading effect and encourages the generated avatars to maintain sharp contour definitions and shadowing consistent with anime aesthetics. Experiments show that AniDream outperforms other frameworks by reducing artifacts and maintaining visual consistency across various poses and viewpoints. It also achieves an average CLIPScore of 33.07, demonstrating its effectiveness in closely aligning generated avatars with text prompts.

**keywords: anime, avatar, diffusion model, generative model, text-to-image, low-rank adaptation, cel-shading.**

## Workflow
AniDream takes a **text prompt** and processes it using a diffusion model guided by skeleton keypoints extracted from **SMPL-X** to ensure anatomical accuracy. It uses a **LoRA** module fine-tuned on anime-style data to adapt the generation style to the anime domain. A **cel-shading-inspired loss** refines the output to produce crisp contours and flat shading consistent with anime aesthetics.

![AniDream Workflow](assets/workflow.png)

## Video
The following video is a compressed preview.  
For the full-resolution supplementary video, please [download it here](assets/demo_video.mp4).

[Watch Compressed Preview](https://github.com/user-attachments/assets/865ca56a-2d88-4c64-986f-a15eb8c8e235)








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
```
We suggest that users follow the installation instructions in [DreamWaltz-G](https://github.com/Yukun-Huang/DreamWaltz-G).

### Human Template Models

Before starting training, please download the human template models from the official project pages:

- [SMPL-X](https://smpl-x.is.tue.mpg.de/)  
- [FLAME](https://flame.is.tue.mpg.de/)  

After downloading, place the files in the following structure:

```
external
└── human_templates
    ├── smplx
    │   ├── SMPLX_NEUTRAL_2020.npz
    │   ├── FLAME_vertex_ids.npy
    │   ├── MANO_vertex_ids.pkl
    │   └── smplx_vert_segmentation.json
    ├── flame
    │   └── FLAME_masks.pkl
    └── vposer
        └── v2.0
            ├── snapshots
            │   ├── V02_05_epoch=08_val_loss=0.03.ckpt
            │   └── V02_05_epoch=13_val_loss=0.03.ckpt
            ├── V02_05.yaml
            └── V02_05.log
```

### Pre-Trained Instant-NGP

Our method builds upon a DreamWaltz-G that adopts a two-stage training pipeline of **NeRF → 3DGS**, where NeRF is initialized with SMPL-X before training.  

DreamWaltz-G provides pre-trained NeRFs available on [HuggingFace](https://huggingface.co/KevinHuang/DreamWaltz-G/tree/main/external/human_templates/instant-ngp).

If you would like to use them, please download and organize the files following the structure below:

```
external
└── human_templates
    ├── instant-ngp
    │   ├── adult_neutral
    │   │   ├── step_005000.pth
    │   │   └── 005000_image.mp4
    ...
```

### Dataset for Inferences

We provide data loaders to read **SMPL-X motion sequences** from four publicly available human motion datasets:

- [Motion-X](https://motion-x-dataset.github.io/)  
- [TalkSHOW](https://talkshow.is.tue.mpg.de/)  
- [AIST++](https://google.github.io/aistplusplus_dataset/)  
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)  
- [Motion-X-ReEnact](https://huggingface.co/KevinHuang/DreamWaltz-G/tree/main/datasets/Motion-X-ReEnact)

These motion data can be used to animate 3D avatars in different demos.  

To use them, please download the datasets from their official websites and organize them as follows (no need to unzip): 

```
datasets
├── 3DPW
│   ├── readme_and_demo.zip
│   ├── sequenceFiles.zip
│   └── SMPL-X.zip
├── AIST++
│   ├── 20210308_cameras.zip
│   └── 20210308_motions.zip
├── Motion-X
│   └── motionx_smplx.zip
├── Motion-X-ReEnact
│   └── Motion-X-ReEnact.zip
└── TalkShow
    ├── chemistry_pkl_tar.tar.gz
    ├── conan_pkl_tar.tar.gz
    └── ...
```


## Usage
Run the following script to generate a single avatar:
```
bash scripts/anime_train.sh "an anime boy"
```

Run the following script to reproduce the experiments presented in our paper:
```
bash scripts/test.sh
```

## Training Stages

#### Canonical NeRF Training – Progressive Low Resolution

The objective is to build a stable canonical NeRF representation of the avatar while gradually increasing image resolution. It follows 64x64 -> 128x128 -> 256x256. This stepwise growth allows the model to refine surface details and textures.



#### Canonical NeRF Training – High Resolution

Refine the avatar at the final high resolution 512x512. Training resumes from the checkpoint of the progressive stage. This stage is computationally heavy and is optional. 

<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/train_low.gif" width="300"/><br/>
      <sub><i>256x256</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/train_high.gif" width="300"/><br/>
      <sub><i>512x512</i></sub>
    </td>
  </tr>
</table>

#### Animatable 3DGS

The animation stages follow the method proposed in DreamWaltz-G.  

Since animation is not the primary focus of our work, this step is provided for completeness but remains optional. 

In the script for training, we provide code for 3D animation using motions from [AIST++](https://google.github.io/aistplusplus_dataset/).


<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/train_rig.gif" width="240"/><br/>
      <sub><i>Canonical Pose Rigging</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/train_random.gif" width="240"/><br/>
      <sub><i>Random Pose Rigging</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/train_aist.gif" width="240"/><br/>
      <sub><i>Animation with AIST++</i></sub>
    </td>
  </tr>
</table>


## Results

Here are example outputs generated by AniDream based on different descriptive text prompts. 

The outputs display diverse anime-style avatars, capturing fine-grained details from each description. 

AniDream can create visually consistent and expressive characters with strong alignment to textual input.

<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/pirate.gif" width="240"/><br/>
      <sub><i>"an anime pirate in a blue and gold coat"</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/glasses.gif" width="240"/><br/>
      <sub><i>"an anime boy with glasses"</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/kimono.gif" width="240"/><br/>
      <sub><i>"an anime girl in a kimono"</i></sub>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/hoodie.gif" width="240"/><br/>
      <sub><i>"an anime boy in a hoodie"</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/jersey.gif" width="240"/><br/>
      <sub><i>"an anime girl in a sports jersey"</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/witch.gif" width="240"/><br/>
      <sub><i>"an anime witch with pointy hat"</i></sub>
    </td>
  </tr>
</table>




### Generation Across Epochs

This animation demonstrates how the generated character evolves during training epochs, showing improvements and refinement in the output quality over time.

<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/animation.gif" width="500"/><br/>
      <sub><i>Avatar Generation</i></sub>
    </td>
  </tr>
</table>


### Normal Map

The following shows a single sample normal map generated for the prompt *"an anime boy"*.  

Normal maps capture detailed surface geometry and shading cues, which are important for realistic lighting and 3D rendering.


<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/boy.gif" width="300"/><br/>
      <sub><i>Avatar</i></sub>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/normal.gif" width="300"/><br/>
      <sub><i>Normal Map</i></sub>
    </td>
  </tr>
</table>

### Animation

Although animation is not the focus of AniDream, the generated avatars can be animated by running the scripts provided by [DreamWaltz-G](https://github.com/Yukun-Huang/DreamWaltz-G). 

This animation shows an avatar performing kung-fu movements. The movements come from an actor in the original video, from which the keypoints are extracted. 

These keypoints are then used to render the avatar, demonstrating how the model animates characters based on real human motion.

<p align="center">
 <img src="assets/animation_video.gif" width="700"/>
</p>

<p align="center" style="font-size: small; margin-top: 10px;">
  The original video sample for this animation comes from Motion-X-ReEnact.
</p>

### 3D Printing

We provide a few simplified avatar models in STL format, ready for 3D printing. 

These files are optimized for standard consumer 3D printers and preserve key avatar features. Feel free to download and print them.

<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/boy_stl.gif" width="300"/><br/>
      <sub><i>"an anime boy"</i></sub><br/>
      <a href="assets/boy.stl" download>Download</a>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/jersey_stl.gif" width="300"/><br/>
      <sub><i>"an anime girl in a sports jersey"</i></sub><br/>
      <a href="assets/jersey.stl" download>Download</a>
    </td>
  </tr>
</table>




## Applications in Industry

While AniDream was initially developed for immersive AR/VR and gaming experiences, its potential extends far beyond these domains. We demonstrate the versatility of AniDream avatars across several core commercial segments of manga and anime production:


- **Manga Panel Generation**  
  A low-resolution render of a generated avatar is processed into black-and-white and passed to GPT-4o with a prompt such as *"Generate a background for this anime art. Like a manga panel."* This results in stylized manga layouts with character integration.

- **Background Animation**  
  Using motion data, the avatars can be animated and composited into anime-style backgrounds. We can control body gestures frame-by-frame, apply real-world motions like dancing or walking, and ensure consistency between static design and animated behavior.

- **Merchandise Production**  
  AniDream avatars can be exported to STL format and 3D printed. This enables direct translation of digital characters into tangible merchandise products, such as action figures.

  
<table align="center">
  <tr>
    <td align="center" style="padding: 10px;">
      <img src="assets/demo_manga.png" height="400"/><br/>
      <sub><i>Manga Panel Generation</i></sub><br/>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/demo_background.gif" height="400"/><br/>
      <sub><i>Background Animation with AIST++</i></sub><br/>
    </td>
    <td align="center" style="padding: 10px;">
      <img src="assets/demo_merchandise.jpg" height="400"/><br/>
      <sub><i>Merchandise Production</i></sub><br/>
    </td>
  </tr>
</table>



These demonstrations highlight the capacity of AniDream to support **cross-medium consistency**, ensuring that the same avatar design can be maintained across visual storytelling, animation, and physical products.

### Supported Content Production Segments

AniDream provides tools and outputs that align with the following industry segments:

- Manga Creation
- Anime-style Animation  
- Merchandise Production  
- Augmented and Virtual Reality  
- Gaming and Interactive Media  
- Metaverse and Social Platforms
---


## Acknowledgments

🔧 This work was supported by the Industrial Technology Innovation Program (Project No. 20023347: *Development of Graph-based Intelligent Metaverse Engine for Immersive Content-sharing Service*) funded by the Ministry of Trade, Industry & Energy of the Republic of Korea.

✨ We gratefully acknowledge the authors of [DreamWaltz-G](https://github.com/Yukun-Huang/DreamWaltz-G). Their framework provided the foundation and inspiration for AniDream, particularly in its skeleton-guided diffusion pipeline and avatar animation capabilities.

## Citation
```
@inproceedings{yamada2025anidream,
  title     = {AniDream: Generating Skeleton-Guided Anime Avatars from Text Prompts},
  author    = {Fernanda Miyuki Yamada and Hiroki Takahashi},
  booktitle = {IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  year      = {2025}
}
```
