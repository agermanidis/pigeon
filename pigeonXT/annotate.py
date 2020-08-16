import random
from collections import defaultdict
import functools
from IPython.display import display, clear_output
from ipywidgets import (
        Button,
        Dropdown,
        HTML,
        HBox,
        VBox,
        IntSlider,
        FloatSlider,
        Textarea,
        Output,
        ToggleButton
)

def annotate(examples,
             task_type='classification',
             options=None,
             shuffle=False,
             include_skip=True,
             include_back=False,
             use_dropdown=False,
             buttons_in_a_row=4,
             reset_buttons_after_click=False,
             example_process_fn=None,
             final_process_fn=None,
             display_fn=display):
    """
    Build an interactive widget for annotating a list of input examples.
    Parameters
    ----------
    examples    : list(any), list of items to annotate
    task_type   : possible options are:
                    - classification
                    - multilabel-classification
                    - captioning
    options     : depending on the task this can be:
                    - list of options
                    - tuple with a range for regression tasks
    shuffle     : bool, shuffle the examples before annotating
    include_skip: bool, include option to skip example while annotating
    use_dropdown: use a dropdown or buttons during classification
    buttons_in_a_row: number of buttons in a row during classification
    reset_buttons_after_click: reset multi-label buttons after each click
    example_process_fn: hooked function to call after each example fn(ix, labels)
    final_process_fn: hooked function to call after annotation is done fn(annotations)
    display_fn  : func, function for displaying an example to the user

    Returns
    -------
    annotations : list of tuples, list of annotated examples (example, label(s))
    """
    examples = list(examples)
    if shuffle:
        random.shuffle(examples)

    annotations = {}
    current_index = -1

    def set_label_text():
        nonlocal count_label
        count_label.value = '{} examples annotated, {} examples left'.format(
            len(annotations), len(examples) - len(annotations)
        )

    def render(index):
        set_label_text()

        for btn in buttons:
            if btn.description == 'prev':
                btn.disabled = index <= 0
            elif btn.description == 'skip':
                btn.disabled = index >= len(examples)
            elif examples[index] in annotations:
                if isinstance(annotations[examples[index]], list):
                    btn.disabled = btn.description in annotations[examples[index]]
                else: 
                    btn.disabled = btn.description == annotations[examples[index]]
                
        if index >= len(examples):
            print('Annotation done.')
            if final_process_fn is not None:
                final_process_fn(list(annotations.items()))
            return
        
        with out:
            clear_output(wait=True)
            display_fn(examples[index])

    def add_annotation(annotation):
        annotations[examples[current_index]] = annotation
        if example_process_fn is not None:
            example_process_fn(examples[current_index], annotation)
        next()

    def next(btn=None):
        nonlocal current_index
        current_index += 1
        render(current_index)

    def prev(btn=None):
        nonlocal current_index
        current_index -= 1
        render(current_index)


    count_label = HTML()
    set_label_text()
    display(count_label)

    buttons = []

    if task_type == 'classification':
        if use_dropdown:
            dd = Dropdown(options=options)
            display(dd)
            btn = Button(description='submit')
            def on_click(btn):
                add_annotation(dd.value)
            btn.on_click(on_click)
            buttons.append(btn)
        else:
            for label in options:
                btn = Button(description=label)
                def on_click(label, btn):
                    add_annotation(label)
                btn.on_click(functools.partial(on_click, label))
                buttons.append(btn)

    elif task_type == 'multilabel-classification':
        for label in options:
            tgl = ToggleButton(description=label)
            buttons.append(tgl)
        btn = Button(description='submit', button_style='info')
        def on_click(btn):
            labels_on = []
            for tgl in buttons:
                if (isinstance(tgl, ToggleButton)):
                    if (tgl.value==True):
                        labels_on.append(tgl.description)
                    if (reset_buttons_after_click):
                        tgl.value=False
            add_annotation(labels_on)
        btn.on_click(on_click)
        buttons.append(btn)

    elif task_type == 'regression':
        target_type = type(options[0])
        if target_type == int:
            cls = IntSlider
        else:
            cls = FloatSlider
        if len(options) == 2:
            min_val, max_val = options
            slider = cls(min=min_val, max=max_val)
        else:
            min_val, max_val, step_val = options
            slider = cls(min=min_val, max=max_val, step=step_val)
        display(slider)
        btn = Button(description='submit', value='submit')
        def on_click(btn):
            add_annotation(slider.value)
        btn.on_click(on_click)
        buttons.append(btn)

    elif task_type == 'captioning':
        ta = Textarea()
        display(ta)
        btn = Button(description='submit')
        def on_click(btn):
            add_annotation(ta.value)
        btn.on_click(on_click)
        buttons.append(btn)
    else:
        raise ValueError('invalid task type')


    if include_back:
        btn = Button(description='prev', button_style='info')
        btn.on_click(prev)
        buttons.append(btn)

    if include_skip:
        btn = Button(description='skip', button_style='info')
        btn.on_click(next)
        buttons.append(btn)

    if len(buttons) > buttons_in_a_row:
        box = VBox([HBox(buttons[x:x + buttons_in_a_row])
                    for x in range(0, len(buttons), buttons_in_a_row)])
    else:
        box = HBox(buttons)

    display(box)

    out = Output()
    display(out)

    next()
    return list(annotations.items())

