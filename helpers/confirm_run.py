import ipywidgets as widgets
from IPython.display import display
import traceback
from time import time

def confirm_run(message, on_confirm):
    """Display a Yes/No confirmation prompt (Jupyter widget) before running a callback.

    Intended for notebook cells where a pipeline step is expensive or
    destructive enough to warrant an explicit click before it executes.

    Args:
        message (str): Text shown above the Yes/No buttons.
        on_confirm (Callable[[], None]): Zero-argument function invoked only if
            the user clicks "Yes". Any exception it raises is caught and its
            traceback printed to the widget output rather than propagating.
    """
    output = widgets.Output()
    _fired = [False]  # guard against multiple clicks before disabled state propagates

    def on_yes(_):
        if _fired[0]:
            return
        _fired[0] = True
        yes_btn.disabled = True
        no_btn.disabled = True
        with output:
            try:
                t = time()
                on_confirm()
                print(f"Done in {time() - t:.1f}s")
            except Exception:
                traceback.print_exc()

    def on_no(_):
        yes_btn.disabled = True
        no_btn.disabled = True
        with output:
            print("Cancelled.")

    yes_btn = widgets.Button(description='Yes', button_style='success')
    no_btn = widgets.Button(description='No', button_style='danger')
    yes_btn.on_click(on_yes)
    no_btn.on_click(on_no)

    display(
        widgets.Label(message),
        widgets.HBox([yes_btn, no_btn]),
        output
    )