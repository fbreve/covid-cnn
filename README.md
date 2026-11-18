# COVID-19 Detection on Chest X-Ray Images: A Comparison of CNN Architectures and Ensembles

This repository contains the code used for the computer simulations reported in the paper:

> Breve, F. A. (2022). COVID-19 detection on Chest X-ray images: A comparison of CNN architectures and ensembles. *Expert Systems with Applications*, **204**, 117549. https://doi.org/10.1016/j.eswa.2022.117549

[Published article](https://doi.org/10.1016/j.eswa.2022.117549) · [Preprint (arXiv)](https://arxiv.org/abs/2111.09972)

## About the study

The study compares 21 convolutional neural network (CNN) architectures for detecting COVID-19 in chest X-ray images, using the COVIDx8B dataset (16,352 images). It also evaluates ensembles of CNN models. Each architecture was trained and evaluated five times using different training/validation splits.

## Dataset and trained weights

The experiments use the **COVIDx8B** chest X-ray dataset. Dataset images are not distributed in this repository; obtain them from the original dataset sources and respect their applicable licenses.

The trained weight files are too large to host in this GitHub repository. The external download link provided for the weights is:

https://1024tera.com/s/1vqm2ha8XENxHkJTzVcTyRA

Availability of this third-party link is not guaranteed.

## Citation

If you use this code in academic work, please cite the published article:

```bibtex
@article{breve2022covid,
  author  = {Breve, Fabricio Aparecido},
  title   = {COVID-19 detection on Chest X-ray images: A comparison of CNN architectures and ensembles},
  journal = {Expert Systems with Applications},
  volume  = {204},
  pages   = {117549},
  year    = {2022},
  doi     = {10.1016/j.eswa.2022.117549}
}
```
