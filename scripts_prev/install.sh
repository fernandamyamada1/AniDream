# Create a new conda environment
conda create -n dreamwaltz2 python=3.11

# Install with conda
## CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cuda==11.8 -c pytorch -c nvidia/label/cuda-11.8.0
conda install fvcore -c conda-forge  # required by pytorch3d
conda install pytorch3d=0.7.5=py311_cu118_pyt210 -c pytorch3d
## CUDA 12.1
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 cuda==12.1 -c pytorch -c nvidia
# conda install -c conda-forge fvcore
# conda install pytorch3d=0.7.5=py311_cu121_pyt210 -c pytorch3d

conda install ninja git-lfs

# Install with pip
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install scikit-image matplotlib imageio plotly open3d trimesh pyrender av decord
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install mediapipe accelerate xatlas libigl
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install pyrallis loguru omegaconf plyfile jaxtyping

/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install -U "huggingface_hub[cli]"

/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/NVlabs/nvdiffrast.git
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/vchoutas/smplx.git
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/nghorbani/human_body_prior.git
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/huggingface/transformers.git
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/huggingface/diffusers.git
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install controlnet-aux==0.0.7

/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/ashawkey/diff-gaussian-rasterization.git
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 -m pip3 install git+https://github.com/graphdeco-inria/gaussian-splatting.git#subdirectory=submodules/simple-knn
