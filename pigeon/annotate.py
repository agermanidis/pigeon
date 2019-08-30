import random
import functools
from IPython.display import display, clear_output
from ipywidgets import Button, Dropdown, HTML, HBox, IntSlider, FloatSlider, Textarea, Output

def annotate(examples,
             options=None,
             shuffle=False,
             include_back=True,
             include_skip=True,
             display_fn=display):
    """
    Build an interactive widget for annotating a list of input examples.

    Parameters
    ----------
    examples: list(any), list of items to annotate
    options: list(any) or tuple(start, end, [step]) or None
             if list: list of labels for binary classification task (Dropdown or Buttons)
             if tuple: range for regression task (IntSlider or FloatSlider)
             if None: arbitrary text input (TextArea)
    shuffle: bool, shuffle the examples before annotating
    include_skip: bool, include option to skip example while annotating
    display_fn: func, function for displaying an example to the user

    Returns
    -------
    annotations : list of tuples, list of annotated examples (example, label)
    """
    examples = list(examples)
    if shuffle:
        random.shuffle(examples)

    annotations = [(ex, None) for ex in examples]
    current_index = -1

    def set_label_text():
        nonlocal count_label
        label_count = len([(x, y) for (x, y) in annotations if y])
        count_label.value = 'Image #{}.  {}/{} examples annotated. {} examples left.'.format(
            current_index, label_count, len(examples), len(examples) - current_index
        )

    # Sync's button style according to the current label
    def update_button_style():
        for btn in buttons:
            if btn.description == annotations[current_index][1]:
                btn.button_style = 'success'
            else:
                btn.button_style = ''

    def show_next():
        nonlocal current_index
        current_index += 1
        set_label_text()
        # If we reach the end, don't update the buttons.
        if current_index < len(examples):
            update_button_style()
        if current_index >= len(examples):
            for btn in buttons:
                if btn.description == 'back':
                    continue
                btn.disabled = True
            print('Annotation done.')
            return
        with out:
            clear_output(wait=True)
            display_fn(examples[current_index])

    def add_annotation(annotation):
        annotations[current_index] = (examples[current_index], annotation)
        show_next()

    def skip(btn):
        show_next()

    def back(btn):
        nonlocal current_index
        # If the annotation was finished. Buttons needs to be re-enabled.
        if current_index >= len(examples):
            for btn in buttons:
                btn.disabled = False
        if current_index is not 0:
            current_index-=2
            show_next()

    count_label = HTML()
    set_label_text()
    display(count_label)

    if type(options) == list:
        task_type = 'classification'
    elif type(options) == tuple and len(options) in [2, 3]:
        task_type = 'regression'
    elif options is None:
        task_type = 'captioning'
    else:
        raise Exception('Invalid options')

    buttons = []

    if include_back:
        btn = Button(description='back')
        btn.on_click(back)
        buttons.append(btn)

    if task_type == 'classification':
        use_dropdown = len(options) > 5

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

    box = HBox(buttons)
    display(box)

    out = Output()
    display(out)

    show_next()

    return annotations
