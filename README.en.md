# Adaptive Conditional Diffusion Sampling with Observations Arriving During Generation

[Русская версия](README.md)

## Materials

- [Abstract](ANNOTATION.en.md)
- [Related work](RELATED_WORK.en.md)
- [Review of 40 sources](LINKREVIEW.en.md)
- [Paper PDF](paper/main_en.pdf)
- [Paper source](paper/main_en.tex)
- [Bibliography](paper/references.bib)

## Abstract

This work investigates ways to adapt diffusion models for trajectory generation to settings where new context arrives during generation. In motion forecasting, additional observations can refine possible futures or substantially change their probabilities, but incorporating these data may require repeated computation that is not always feasible under limited time and computational budgets. I investigate how to update a trajectory being generated using new data, when previous computation can be reused, and how the update strategy affects forecast quality and computational cost. To this end, I will review existing approaches to diffusion forecasting, streaming generation, and conditional sampling, compare their assumptions and limitations, and identify strategies applicable to context changes within an unfinished generation process. The comparison will consider different observation arrival times, different degrees of influence on the forecast, and comparable computational budgets, assessing trajectory accuracy, diversity of possible outcomes, and the time required to produce an updated result. I will also examine probabilistic forecast quality, since a single generated trajectory being close to the observed motion does not necessarily mean that the model correctly represents uncertainty about the future. Based on the identified limitations, I will propose possible improvements and test the conditions under which they make more effective use of new observations. The investigation aims to establish the capabilities and limits of adapting diffusion sampling to changing context under limited computational resources.

**Keywords:** diffusion models; trajectory generation; probabilistic forecasting; conditional sampling; streaming observations; computational efficiency.

## Research status

The study is at the literature review and problem refinement stage. No particular method has yet been selected, and no experimental results are claimed. The focus is on new observations arriving while a trajectory forecast is being computed. Author and advisor details will be added after confirmation.

## Building the PDFs

The PDFs were built from LaTeX sources using Tectonic 0.17.0. On its first run, Tectonic downloads the required packages and Computer Modern Unicode fonts. To build both language versions, run:

```sh
cd paper
tectonic main_ru.tex
tectonic main_en.tex
```
