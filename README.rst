🐦 pigeonML - Quickly annotate data on Jupyter
========================

Pigeon is a simple widget that lets you quickly annotate a dataset of
unlabeled examples from the comfort of your Jupyter notebook.

Pigeon currently supports annotation for classification tasks (set of
labels), regression tasks (int/float range), or captioning tasks
(variable-length text). Anything that can be displayed on Jupyter
(text, images, audio, graphs, etc.) can be displayed by pigeon
by providing the appropriate :code:`display_fn` argument.

Installation
-----

.. code-block:: bash

    pip install pigeon-jupyter

Examples
-----

- Text classification

Code:

.. code-block:: python

    from pigeon import annotate
    annotations = annotate(
      ['I love this movie', 'I was really disappointed by the book'],
      options=['positive', 'negative']
    )


Preview:

.. image:: http://i.imgur.com/00ry4Li.gif

- Image classification

Code:

.. code-block:: python

    from pigeon import annotate
    from IPython.display import display, Image

    annotations = annotate(
      ['assets/img_example1.jpg', 'assets/img_example2.jpg'],
      options=['cat', 'dog', 'horse'],
      display_fn=lambda filename: display(Image(filename))
    )

Preview:

.. image:: http://i.imgur.com/PiE3eDt.gif
