# Адаптивное условное диффузионное сэмплирование при поступлении новых наблюдений в процессе генерации

Adaptive Conditional Diffusion Sampling with Observations Arriving During Generation

## Материалы / Assets

- [Аннотация на русском и английском / Russian and English abstracts](ANNOTATION.md)
- [Обзор литературы / Related Work](RELATED_WORK.md)
- [Анализ 40 источников / Review of 40 sources](LINKREVIEW.md)
- [Статья в PDF / Paper PDF](paper/main.pdf)
- [Исходник статьи / Paper source](paper/main.tex)
- [Библиография / Bibliography](paper/references.bib)

## Аннотация

Данная работа посвящена исследованию способов адаптации диффузионных моделей для генерации траекторий к условиям, когда новый контекст поступает во время генерации. В задачах прогнозирования движения дополнительные наблюдения могут уточнить возможные варианты будущего или существенно изменить их вероятности, однако учёт этих данных может потребовать повторных вычислений, которые не всегда допустимы при ограниченном времени и вычислительном бюджете. Я исследую, как обновлять формируемую траекторию с учётом новых данных, в каких случаях можно использовать уже выполненные вычисления и как выбранный способ обновления влияет на качество прогноза и затраты на его получение. Для этого я рассмотрю существующие подходы к диффузионному прогнозированию, потоковой генерации и условному сэмплированию, сопоставлю их предположения и ограничения и выделю способы, применимые к изменению контекста внутри незавершённого процесса генерации. Сравнение предполагается проводить при разных моментах поступления наблюдений, разной степени их влияния на прогноз и сопоставимых вычислительных бюджетах, учитывая точность траекторий, разнообразие возможных исходов и время получения обновлённого результата. При этом я отдельно рассмотрю качество вероятностного прогноза, поскольку близость одной сгенерированной траектории к наблюдаемому движению ещё не означает, что модель корректно отражает неопределённость будущего. На основе выявленных ограничений я предложу возможные способы улучшения существующих подходов и проверю, в каких условиях они позволяют эффективнее использовать новые наблюдения. Исследование направлено на определение возможностей и границ адаптации диффузионного сэмплирования к изменяющемуся контексту при ограниченных вычислительных ресурсах.

## Abstract

This work investigates ways to adapt diffusion models for trajectory generation to settings where new context arrives during generation. In motion forecasting, additional observations can refine possible futures or substantially change their probabilities, but incorporating these data may require repeated computation that is not always feasible under limited time and computational budgets. I investigate how to update a trajectory being generated using new data, when previous computation can be reused, and how the update strategy affects forecast quality and computational cost. To this end, I will review existing approaches to diffusion forecasting, streaming generation, and conditional sampling, compare their assumptions and limitations, and identify strategies applicable to context changes within an unfinished generation process. The comparison will consider different observation arrival times, different degrees of influence on the forecast, and comparable computational budgets, assessing trajectory accuracy, diversity of possible outcomes, and the time required to produce an updated result. I will also examine probabilistic forecast quality, since a single generated trajectory being close to the observed motion does not necessarily mean that the model correctly represents uncertainty about the future. Based on the identified limitations, I will propose possible improvements and test the conditions under which they make more effective use of new observations. The investigation aims to establish the capabilities and limits of adapting diffusion sampling to changing context under limited computational resources.

## Статус исследования / Research status

Исследование находится на этапе анализа литературы и уточнения постановки. Конкретный метод ещё не выбран; экспериментальные результаты не заявляются. Основной предмет — поступление новых наблюдений во время вычисления прогноза траектории. Авторские данные и сведения о научном руководителе будут добавлены после уточнения.

The study is at the literature review and problem refinement stage. No particular method has yet been selected, and no experimental results are claimed. The focus is on new observations arriving while a trajectory forecast is being computed. Author and advisor details will be added after confirmation.

## Сборка PDF / Building the PDF

Оформление основано на [шаблоне курса](https://github.com/kisnikser/m1p-template/tree/main/paper). Для сборки используется Tectonic 0.17.0; при первом запуске он загружает необходимые LaTeX-пакеты и шрифты Computer Modern Unicode.

The layout follows the [course template](https://github.com/kisnikser/m1p-template/tree/main/paper). The document is built with Tectonic 0.17.0, which downloads the required LaTeX packages and Computer Modern Unicode fonts on its first run.

```sh
cd paper
tectonic main.tex
```
