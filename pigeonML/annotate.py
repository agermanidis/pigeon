import random
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
             options=None,
             task_type='classification',
             shuffle=False,
             include_skip=True,
             use_dropdown_in_classification=False,
             buttons_in_a_row=4,
             reset_buttons_after_click=False,
             display_fn=display):
    """
    Build an interactive widget for annotating a list of input examples.

    Parameters
    ----------
    examples : list(any), list of items to annotate
    options  : list(any) or tuple(start, end, [step]) or None
               if list  : list of labels for multiclass or multilabel classification tasks
               if tuple : range for regression task (IntSlider or FloatSlider)
               if None  : arbitrary text input (TextArea)
    task_type: str, identifier for the annotation task
               'classification': multi class classification
               'multi-label-classification': multi label classification
               'regression': regression type classification
               'arbitrary': arbitrary text input (captioning)

    Optional parameters
    ----------
    shuffle                        : bool, shuffle the examples before annotating
    include_skip                   : bool, include option to skip example while annotating
    use_dropdown_in_classification : bool, use dropdown instead of buttons
    buttons_in_a_row               : int,  amount of buttons per row
    reset_buttons_after_click      : bool, reset options after each submit
    display_fn                     : func, function for displaying an example to the user

    Returns
    -------
    annotations : list of tuples, list of annotated examples (example, label)

    """
    examples = list(examples)
    if shuffle:
        random.shuffle(examples)

    annotations = []
    current_index = -1

    def set_label_text():
        nonlocal count_label
        count_label.value = '{} examples annotated, {} examples left'.format(
            len(annotations), len(examples) - current_index
        )

    def show_next():
        nonlocal current_index
        current_index += 1
        set_label_text()
        if current_index >= len(examples):
            for btn in buttons:
                btn.disabled = True
            print('Annotation done.')
            return
        with out:
            clear_output(wait=True)
            display_fn(examples[current_index])

    def add_annotation(annotation):
        annotations.append((examples[current_index], annotation))
        show_next()

    def skip(btn):
        show_next()

    count_label = HTML()
    set_label_text()
    display(count_label)

    buttons = []

    if task_type == 'classification':
        if use_dropdown_in_classification:
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
        btn = Button(description='submit')
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
        btn = Button(description='submit')
        def on_click(btn):
            add_annotation(slider.value)
        btn.on_click(on_click)
        buttons.append(btn)

    else:
        ta = Textarea()
        display(ta)
        btn = Button(description='submit')
        def on_click(btn):
            add_annotation(ta.value)
        btn.on_click(on_click)
        buttons.append(btn)

    if include_skip:
        btn = Button(description='skip')
        btn.on_click(skip)
        buttons.append(btn)
    if len(buttons) > buttons_in_a_row:
        box = VBox([HBox(buttons[x:x + buttons_in_a_row])
                    for x in range(0, len(buttons), buttons_in_a_row)])
    else:
        box = HBox(buttons)

    display(box)

    out = Output()
    display(out)

    show_next()

    return annotations

