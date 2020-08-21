# 🐦 pigeonXT - Quickly annotate data in Jupyter Lab
PigeonXT is an extention to the original [Pigeon](https://github.com/agermanidis/pigeon), created by [Anastasis Germanidis](https://pypi.org/user/agermanidis/).
PigeonXT is a simple widget that lets you quickly annotate a dataset of
unlabeled examples from the comfort of your Jupyter notebook.

PigeonXT currently support the following annotation tasks:
- binary / multi-class classification
- multi-label classification
- regression tasks
- captioning tasks

Anything that can be displayed on Jupyter
(text, images, audio, graphs, etc.) can be displayed by pigeon
by providing the appropriate `display_fn` argument.

Additionally, custom hooks can be attached to each row update (`example_process_fn`),
or when the annotating task is complete(`final_process_fn`).

There is a full blog post on the usage of PigeonXT on [Towards Data Science](https://towardsdatascience.com/quickly-label-data-in-jupyter-lab-999e7e455e9e).

### Contributors
- Anastasis Germanidis
- Dennis Bakhuis
- Ritesh Agrawal

## Installation
PigeonXT obviously needs a Jupyter Lab environment. Futhermore, it requires ipywidgets.
The widget itself can be installed using pip:
```bash
    pip install pigeonXT-jupyter
```

To run the provided examples in a new environment using Conda:
```bash
    conda create --name pigeon python=3.7
    conda activate pigeon
    conda install nodejs
    pip install numpy pandas jupyterlab ipywidgets
    jupyter nbextension enable --py widgetsnbextension
    jupyter labextension install @jupyter-widgets/jupyterlab-manager

    pip install pigeonXT-jupyter
```

Starting Jupyter Lab environment:
```bash
    jupyter lab
```

## Examples
Examples are also provided in the accompanying notebook.

### Binary or multi-class text classification
Code:
```python
    from pigeonXT import annotate
    annotations = annotate(
      ['I love this movie', 'I was really disappointed by the book'],
      options=['positive', 'negative', 'inbetween']
    )
```

Preview:
![Jupyter notebook multi-class classification](/assets/multiclassexample.png)

### Multi-label text classification
Code:
```python
    from pigeonXT import annotate
    import pandas as pd

    df = pd.DataFrame([
        {'title': 'Star wars'},
        {'title': 'The Positively True Adventures of the Alleged Texas Cheerleader-Murdering Mom'},
        {'title': 'Eternal Sunshine of the Spotless Mind'},
        {'title': 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb'},
        {'title': 'Killer klowns from outer space'},
    ])

    labels = ['Adventure', 'Romance', 'Fantasy', 'Science fiction', 'Horror', 'Thriller']

    annotations = annotate( df.title,
                          options=labels,
                          task_type='multilabel-classification',
                          buttons_in_a_row=3,
                          reset_buttons_after_click=True,
                          include_skip=True)
```

Preview:
![Jupyter notebook multi-label classification](/assets/multilabelexample.png)

### Image classification
Code:
```python
    from pigeonXT import annotate
    from IPython.display import display, Image

    annotations = annotate(
      ['assets/img_example1.jpg', 'assets/img_example2.jpg'],
      options=['cat', 'dog', 'horse'],
      display_fn=lambda filename: display(Image(filename))
    )
```

Preview:
![Jupyter notebook multi-label classification](/assets/imagelabelexample.png)

### multi-label text classification with custom hooks
Code:
```python
    from pigeonXT import annotate
    import pandas as pd
    import numpy as np
    from pathlib import Path


    df = pd.DataFrame([
        {'title': 'Star wars'},
        {'title': 'The Positively True Adventures of the Alleged Texas Cheerleader-Murdering Mom'},
        {'title': 'Eternal Sunshine of the Spotless Mind'},
        {'title': 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb'},
        {'title': 'Killer klowns from outer space'},
    ])

    labels = ['Adventure', 'Romance', 'Fantasy', 'Science fiction', 'Horror', 'Thriller']
    shortLabels = ['A', 'R', 'F', 'SF', 'H', 'T']

    df.to_csv('inputtestdata.csv', index=False)


    def setLabels(labels, numClasses):
        row = np.zeros([numClasses], dtype=np.uint8)
        row[labels] = 1
        return row

    def labelPortion(inputFile,
                     labels = ['yes', 'no'],
                     outputFile='output.csv',
                     portionSize=2,
                     textColumn='title',
                     shortLabels=None):
        if shortLabels == None:
            shortLabels = labels
        out = Path(outputFile)
        if out.exists():
            outdf = pd.read_csv(out)
            currentId = outdf.index.max() + 1
        else:
            currentId = 0
        indf = pd.read_csv(inputFile)
        examplesInFile = len(indf)
        indf = indf.loc[currentId:currentId + portionSize - 1]
        actualPortionSize = len(indf)
        print(f'{currentId + 1} - {currentId + actualPortionSize} of {examplesInFile}')
        sentences = indf[textColumn].tolist()

        for label in shortLabels:
            indf[label] = None

        def updateRow(title, selectedLabels):
            print(title, selectedLabels)
            labs = setLabels([labels.index(y) for y in selectedLabels], len(labels))
            indf.loc[indf.title == title, shortLabels] = labs

        def finalProcessing(annotations):
            if out.exists():
                prevdata = pd.read_csv(out)
                outdata = pd.concat([prevdata, indf]).reset_index(drop=True)
            else:
                outdata = indf.copy()
            outdata.to_csv(out, index=False)

        annotated = annotate( sentences,
                              options=labels,
                              task_type='multilabel-classification',
                              buttons_in_a_row=3,
                              reset_buttons_after_click=True,
                              include_skip=False,
                              example_process_fn=updateRow,
                              final_process_fn=finalProcessing)
        return indf


    annotations = labelPortion('inputtestdata.csv',
                               labels=labels,
                               shortLabels= shortLabels)
```

Preview:
![Jupyter notebook multi-label classification](/assets/pigeonhookfunctions.png)


The complete and runnable examples are available in the provided Notebook.

