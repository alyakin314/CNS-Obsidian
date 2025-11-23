import ipywidgets as widgets
from IPython.display import display


class ImageLabelingApp:
    def __init__(self, dataset, captions):
        self.dataset = dataset
        self.captions = captions
        self.numbers = []
        self.current_index = 0
        self.output = widgets.Output()

        # Create the buttons
        self.button_imaging = widgets.Button(description="Imaging")
        self.button_imaging.on_click(lambda b: self.on_button_click("Imaging"))

        self.button_interesting = widgets.Button(
            description="Not imaging but interesting"
        )
        self.button_interesting.on_click(
            lambda b: self.on_button_click("Not imaging but interesting")
        )

        self.button_discard = widgets.Button(description="Discard")
        self.button_discard.on_click(lambda b: self.on_button_click("Discard"))

    def on_button_click(self, option):
        number = (
            2
            if option == "Imaging"
            else 1
            if option == "Not imaging but interesting"
            else 0
        )
        self.numbers.append(number)
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.display_item()
        else:
            self.display_final()

    def display_item(self):
        self.output.clear_output()
        with self.output:
            display(self.dataset[self.current_index])
            display(
                widgets.HBox(
                    [
                        self.button_imaging,
                        self.button_interesting,
                        self.button_discard,
                    ]
                )
            )
            print(self.captions[self.current_index])

    def display_final(self):
        self.output.clear_output()
        with self.output:
            display(widgets.HTML("<h3>Finished!</h3>"))
            display(widgets.HTML(f"Numbers: {self.numbers}"))

    def start(self):
        # Display the first item
        self.display_item()
        # Display the output area
        display(self.output)

    def get_results(self):
        return self.numbers
