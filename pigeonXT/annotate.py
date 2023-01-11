import random
from collections import defaultdict
import functools
import warnings
from IPython.display import display, HTML, clear_output
from ipywidgets import (
        Button,
        Dropdown,
        HTML,
        HBox,
        VBox,
        IntSlider,
        FloatSlider,
        Layout,
        Textarea,
        Output,
        ToggleButton
)
import pandas as pd
import re



def annotate(
    examples,
    task_type='classification',
    options=None,
    shuffle=False,
    
    include_next=True,
    include_back=True,
    stop_at_last_example=True,
    use_dropdown=False,
    buttons_in_a_row=4,
    reset_buttons_after_click=True,
    
    example_process_fn=None,
    final_process_fn=None,
    display_fn=display,
    
    example_column='example',
    value_column='label',
    id_column='id',
    return_type='dataframe',
    bionic_reading=False,
    checkerboard=False
):
    """
    Build an interactive widget for annotating a list or DataFrame of input examples.
    
    Parameters
    ----------
    examples    : list or pandas.DataFrame
                    This can be a list of any type or a Pandas dataframe.
                    
                    If using a DataFrame, make sure that at least the example_column
                    is set to the column holding the examples.
                    
                    When a DataFrame of a previous annotation is provided, it can
                    be relabeled and shows previous selected labels.
    task_type   : str
                    Possible options are:
                    - classification
                    - multilabel-classification
                    - regression
                    - captioning
    options     : list, tuple, or None
                    Depending on the task this can be:
                    - list of str for (multilabel) classification.
                    - tuple with a range for regression tasks
                    - None (actually ignored) for captioning
                    
    shuffle                   : bool, shuffle the examples before annotating
    include_skip              : bool, include option to skip example while annotating
    include_back              : bool, include option to navigate to previous example
    use_dropdown              : bool, use a dropdown or buttons during classification
    buttons_in_a_row          : int, number of buttons in a row during classification
    reset_buttons_after_click : bool, reset multi-label buttons after each click
    
    example_process_fn : function, hooked function to call after each example fn(ix, labels)
    final_process_fn   : function, hooked function to call after annotation is done fn(annotations)
    display_fn         : function, function for displaying an example to the user
                          Default, it uses the IPython display function    
    example_column : str, column name which holds all examples. Required when using DataFrame
    value_column   : str
                        column to store the result for classification (not for multilabel), regression,
                        captioning. For multilabel, each option will be a column in the dataframe.
    id_column      : str, optional
                        Column name which holds the id of the example. If available, will be shown in
                        the progress row.
    return_type    : str, 'dataframe' or 'dict'
                        By default, annotate will return a DataFrame with the annotations. For compatability,
                        when return_type is 'dict' it can also return a dictionary with changed annotations.
    
    checkerboard : displays words in checkerboard pattern
    bionic_reading : at boldfacing for readability

    Returns
    -------
    annotations : pandas.DataFrame or dict
                    Depending on return_type it will return a DataFrame (preferred) or a dict with the
                    annotations.
                    
                    The dict will have the form: {example: label} and only return the labeled examples that
                    are changed (using the submit button).
                    
                    The DataFrame will have a column with the examples and if it is multilabel, a column for
                    each label. For regular classification, the labeled values are in the value_column. When
                    a DataFrame is used as input, all other columns such as id, are kept intact.
    """
    # Parameter checkes
    task_type = task_type.lower()
    if task_type not in [
        'classification',
        'multilabel-classification',
        'regression',
        'captioning',
    ]:
        raise ValueError("task_type should be 'classification', 'multilabel-classification', 'regression', or 'captioning'")
        
    if not isinstance(examples, (list, pd.DataFrame)):
        raise TypeError('examples should be of type list or pandas.DataFrame')

    if task_type == 'regression':
        if not isinstance(options, tuple):
            raise TypeError('options should be of type tuple for regression tasks')
        if len(options) != 2 and len(options) != 3:
            raise ValueError('options should be a tuple (min, max) or (min, max, step)')

    if task_type in ['multilabel-classification', 'classification']:
        if not isinstance(options, list):
            raise TypeError('options should be of type list for classification tasks')
    
    return_type = return_type.lower()
    if return_type not in ['dataframe', 'dict']:
        raise TypeError("return_type should be 'dataframe' or 'dict'")        
    
    # create annotations object as DataFrame
    if isinstance(examples, pd.DataFrame):
        # Examples is a dataframe
        annotations = examples.copy()
        annotations['changed'] = False 
    else:
        # Examples is a list
        annotations = pd.DataFrame({
            example_column: examples,
            'changed': False,
        })
    
    # add options as columns
    if isinstance(options, list):
        # list of labels
        if task_type == 'classification':
            annotations[value_column] = ''
        else:
            for label in options:
                if label not in annotations.columns:
                    annotations[label] = False
    elif isinstance(options, tuple):
        # regression
        if value_column not in annotations.columns:
            annotations[value_column] = 0
    else:
        # captioning
        if value_column not in annotations.columns:
            annotations[value_column] = ''
            
    # shuffle if needed
    if shuffle:
        annotations = annotations.sample(frac=1).copy()
      
    current_index = -1
    annotations_dict = {}

    def set_label_text(index):
        """
        Create info string with annotation progress
        """
        nonlocal count_label
        labeled = len(annotations.loc[annotations['changed']])
        str_output =  f'{labeled} of {len(annotations)} Examples annotated, Current Position: {index + 1} '
        if id_column in annotations.columns and index >= 0 and index < len(annotations):
            ix = annotations.iloc[index].name
            str_output += f"(id: {annotations.at[ix, id_column]}) "
        count_label.value = str_output

    def render(index):
        """
        Render current index of the annotation
        """
        set_label_text(index)
        if index >= len(annotations):
            if stop_at_last_example:
                print('Annotation done.')
                if final_process_fn is not None:
                    final_process_fn(annotations)
                for button in buttons:
                    button.disabled = True
                set_label_text(index - 1)
            else:
                prev_example()
            return
        # render buttons
        ix = annotations.iloc[index].name
        for button in buttons:
            if button.description == 'prev':
                # disable previous button when at first example
                button.disabled = index <= 0
            elif button.description == 'next':
                # disable skip button when at last example
                button.disabled = index >= len(annotations) - 1
            elif button.description != 'submit':
                if task_type == 'classification':
                    if annotations.at[ix, value_column] == button.description:
                        button.icon = 'check'
                    else:
                        button.icon = ''
                elif task_type == 'multilabel-classification':
                    button.value = bool(annotations.at[ix, button.description])
        # render dropdown
        if use_dropdown:
            current_value = annotations.at[ix, value_column]
            if current_value in dd.options:
                dd.value = current_value
        # slider while regression
        if task_type == 'regression':
            slider.value = annotations.at[ix, value_column]
        # captioning
        if task_type == 'captioning':
            ta.value = annotations.at[ix, value_column]

        # display new example
        with out:
            clear_output(wait=True)
            example_text = annotations.at[ix, example_column]
            style_text = "<style>div.output_scroll { height: 88em; width: 100%}</style>"

            if bionic_reading:
                min_word_bion = 6
                first_bion_chars = 2
                example_text = " ".join([f'<b>{word[:first_bion_chars]}</b>{word[first_bion_chars:]}' 
                    if len(word) > min_word_bion else word for word in example_text.split()])

            if checkerboard:
                colDict = {}
                colDict[0] = '#e1e3e1' 
                colDict[1] = '#bab8b8'
                example_text = ".".join([f'<span style="background-color:{colDict[k%2]}">{word}</span>' 
                    for k,word in enumerate(example_text.split("."))])

            decorated_text = HTML(style_text+f"<body><div>{example_text}</div></body>")

            display_fn(decorated_text);

    def add_annotation(annotation):
        """
        Toggle annotation
        """
        if return_type == 'dict':
            annotations_dict[annotations.at[current_index, example_column]] = annotation
        ix = annotations.iloc[current_index].name
        if task_type == 'multilabel-classification':
            for label in options:
                annotations.at[ix, label] = label in annotation
        else:
            # multi-class, regression, captioning
            annotations.at[ix, value_column] = annotation
        annotations.at[ix, 'changed'] = True
        if example_process_fn is not None:
            example_process_fn(annotations.at[ix, example_column], annotation)
        next_example()

    def next_example(button=None):
        """
        Increase current index
        """
        nonlocal current_index
        if current_index < len(annotations):
            current_index += 1
            render(current_index)

    def prev_example(button=None):
        """
        Decrease current index
        """
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            render(current_index)

    count_label = HTML()
    set_label_text(current_index)
    display(count_label)
    buttons = []
    
    if task_type == 'classification':
        if use_dropdown:
            dd = Dropdown(options=options)
            display(dd)
            btn = Button(description='submit')

            def on_click(button):
                add_annotation(dd.value)

            btn.on_click(on_click)
            buttons.append(btn)
        else:
            for label in options:
                btn = Button(description=label)

                def on_click(lbl, button):
                    add_annotation(lbl)

                btn.on_click(functools.partial(on_click, label))
                buttons.append(btn)
            
    elif task_type == 'multilabel-classification':
        for label in options:
            tgl = ToggleButton(description=label)
            buttons.append(tgl)
        btn = Button(description='submit', button_style='info')

        def on_click(button):
            labels_on = []
            for tgl_btn in buttons:
                if isinstance(tgl_btn, ToggleButton):
                    if tgl_btn.value:
                        labels_on.append(tgl_btn.description)
                    if reset_buttons_after_click:
                        tgl_btn.value = False
            add_annotation(labels_on)

        btn.on_click(on_click)
        buttons.append(btn)
        
    elif task_type == 'regression':
        # check if tuple is int or float
        target_type = type(options[0])
        if target_type == int:
            cls = IntSlider
        else:
            cls = FloatSlider

        # create slider
        if len(options) == 2:
            min_val, max_val = options
            slider = cls(min=min_val, max=max_val)
        else:
            min_val, max_val, step_val = options
            slider = cls(min=min_val, max=max_val, step=step_val)
        display(slider)

        # submit button
        btn = Button(description='submit', value='submit')
        def on_click(button):
            add_annotation(slider.value)
        btn.on_click(on_click)
        buttons.append(btn)

    elif task_type == 'captioning':
        ta = Textarea()
        display(ta)
        btn = Button(description='submit')

        def on_click(button):
            add_annotation(ta.value)

        btn.on_click(on_click)
        buttons.append(btn)
    else:
        raise ValueError('invalid task type')

    if include_back:
        btn = Button(description='prev', button_style='info')
        btn.on_click(prev_example)
        buttons.append(btn)

    if include_next:
        btn = Button(description='next', button_style='info')
        btn.on_click(next_example)
        buttons.append(btn)

    if len(buttons) > buttons_in_a_row:
        box = VBox([HBox(buttons[x:x + buttons_in_a_row])
                    for x in range(0, len(buttons), buttons_in_a_row)])
    else:
        box = HBox(buttons)

    display(box)

    out = Output()
    display(out)

    next_example()
    
    # return object
    if return_type == 'dataframe':
        return annotations
    else:
        return annotations_dict
