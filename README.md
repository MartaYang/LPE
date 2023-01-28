<div align="center">
  <h1>Semantic Guided Latent Parts Embedding for Few-Shot Learning <br> (WACV 2023)</h1>
</div>

<div align="center">
  <h3><a href=https://martayang.github.io/>Fengyuan Yang</a>, <a href=https://vipl.ict.ac.cn/homepage/rpwang/index.htm>Ruiping Wang</a>, <a href=http://people.ucas.ac.cn/~xlchen?language=en>Xilin Chen</a></h3>
</div>

<div align="center">
  <h4> <a href=https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Semantic_Guided_Latent_Parts_Embedding_for_Few-Shot_Learning_WACV_2023_paper.pdf>[Paper link]</a>, <a href=https://openaccess.thecvf.com/content/WACV2023/supplemental/Yang_Semantic_Guided_Latent_WACV_2023_supplemental.pdf>[Supp link]</a></h4>
</div>

## 1. Requirements
* Python 3.7
* PyTorch 1.9.0


## 2. Datasets

* Original datasets
    * All 4 datasets are the same as previous works (e.g., [DeepEMD](https://github.com/icoz69/DeepEMD), [renet](https://github.com/dahyun-kang/renet)), and can be download from their links: [miniImagenet](https://drive.google.com/file/d/191cFzwwNTzG_mHUDABF0Nh77cI6pa-qq/view?usp=sharing), [tieredImageNet](https://drive.google.com/file/d/1ANczVwnI1BDHIF65TgulaGALFnXBvRfs/view?usp=sharing), [CIFAR-FS](https://drive.google.com/file/d/1Lq2USoQmbFgCFJlGx3huFSfjqwtxMCL8/view?usp=sharing), [CUB-FS](https://drive.google.com/file/d/1B8jmZin9teye7Lte9ZKsQ3lyMASbxune/view?usp=sharing). 
    * Download and extract them in a certain folder, let's say  `/data/FSLDatasets/LPE_dataset`, then remember to set `args.data_dir` to this folder when running the code later.

* Semantic embeddings
    * **Additional semantic embeddings** of these 4 datasets leveraged by our method can be downloaded [here](https://drive.google.com/drive/folders/1fXpmCU3y5ajKJtDdvjUNRTl5UyHvmyKM?usp=sharing). 
    * Download and put them in the corresponding dataset folder (e.g., put `miniimagenet/wnid2CLIPemb_zscore.npy` to `/data/FSLDatasets/LPE_dataset/miniimagenet/wnid2CLIPemb_zscore.npy`), then remember to set `args.semantic_path` to the location of this file and `args.sem_dim` accordingly when running the code later.

## 3. Usage

Our training and testing scripts are all at `scripts/train.sh`, and corresponding output logs can found at this folder too.

## 4. Results

The 1-shot and 5-shot classification results can be found in the corresponding output logs.

## Citation

If you find our paper or codes useful, please consider citing our paper:

```bibtex
@InProceedings{Yang_2023_WACV,
    author    = {Yang, Fengyuan and Wang, Ruiping and Chen, Xilin},
    title     = {Semantic Guided Latent Parts Embedding for Few-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5447-5457}
}
```

## Acknowledgments

Our codes are based on [renet](https://github.com/dahyun-kang/renet) and [DeepEMD](https://github.com/icoz69/DeepEMD), and we really appreciate it. 

## Further

If you have any question, feel free to contact me. My email is _fengyuan.yang@vipl.ict.ac.cn_