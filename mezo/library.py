"""
library.py - Mezo module script for managing the samples library in the Mezo application.

This script provides functionalities for managing and interacting with the samples library in
the Mezo application. It includes functions to retrieve sample data from a SQLite database,
display samples in a user interface, and handle user interactions for creating new samples.

Key Functions:
- get_samples_overview: Retrieves an overview of all samples stored in the database.
- get_id_from_name: Fetches the sample ID corresponding to a given sample name.
- open_editor: Opens the editor page for a selected sample.
- init_samples_library: Initializes the samples library view with sample cards.
- library: Creates the main view for the samples library, including controls for adding new samples.

Constants:
- COLOR_SCHEME: A dictionary defining color codes for various UI elements.
- DATA_DIR: Directory path for storing sample data.
- MEZO_DB: Database filename for storing sample information.
- SVG: A string containing SVG markup for visual representation in the UI.

Author: Roman Kozlov
Github: https://github.com/donatorex
"""

import os
import sqlite3
from datetime import datetime
from typing import Union

import flet as ft
from PIL import Image

COLOR_SCHEME = {
    'primary': '#C3C7CF',
    'secondary': '#535353',
    'highlight': '#835AFF',
    'background': '#282828'
}

DATA_DIR = 'data'
MEZO_DB = 'mezo.db'

SVG = """
<svg width="400" height="360" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <linearGradient y2="0" x2="1" y1="1" x1="0" id="svg_8">
   <stop offset="0.39063" stop-opacity="0.99219" stop-color="#ecffb5"/>
   <stop offset="1" stop-opacity="0.99609" stop-color="#bfff00"/>
   <stop offset="NaN" stop-opacity="0" stop-color="0"/>
   <stop offset="NaN" stop-opacity="0" stop-color="0"/>
   <stop offset="NaN" stop-opacity="0" stop-color="0"/>
   <stop offset="NaN" stop-opacity="0" stop-color="0"/>
   <stop offset="NaN" stop-opacity="0" stop-color="0"/>
  </linearGradient>
 </defs>
 <g>
  <title>Слой 1</title>
  <ellipse stroke-width="5" ry="146.5" rx="146.5" id="svg_1" cy="157.25579" cx="157.01092" stroke="#b1e800" fill="url(#svg_8)"/>
  <ellipse stroke="#b1e800" stroke-width="0" ry="3.6" rx="3.6" id="svg_9" cy="156.97843" cx="157.95536" fill="#b1e800"/>
  <g id="svg_28">
   <ellipse stroke="#b1e800" stroke-width="5" ry="56.05556" rx="56.05556" id="svg_12" cy="294.3229" cx="332.7887" fill="url(#svg_8)"/>
   <ellipse stroke="#b1e800" stroke-width="0" ry="3.6" rx="3.6" id="svg_25" cy="294" cx="333" fill="#b1e800"/>
  </g>
  <g id="svg_29">
   <ellipse stroke="#b1e800" stroke-width="5" ry="23.16666" rx="23.16666" id="svg_21" cy="321.43405" cx="57.6776" fill="url(#svg_8)"/>
   <ellipse stroke="#b1e800" stroke-width="0" ry="3.6" rx="3.6" id="svg_26" cy="321" cx="58" fill="#b1e800"/>
  </g>
  <g id="svg_30">
   <ellipse stroke="#b1e800" stroke-width="5" ry="29.75203" rx="28.04471" id="svg_15" cy="65.91705" cx="340.53938" fill="url(#svg_8)"/>
   <ellipse stroke="#b1e800" stroke-width="0" ry="3.6" rx="3.6" id="svg_27" cy="66" cx="341" fill="#b1e800"/>
  </g>
 </g>
</svg>
"""  # noqa: E501


def get_samples_overview() -> list:
    """
    Get overview of all samples in the database.

    Returns a list of tuples, where each tuple represents a sample. The tuple contains
    the sample ID, creation timestamp, name of the sample, description and count of sample images.

    :return: list - List of tuples, where each tuple contains sample parameters.
    """
    conn = sqlite3.connect('mezo.db')
    cur = conn.cursor()
    try:
        cur.execute('SELECT * FROM samples')
        samples = cur.fetchall()
        return samples
    except sqlite3.Error as e:
        print(f"Error while getting samples overview: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def get_id_from_name(name: str) -> Union[int, None]:
    """
    Retrieve the sample_id associated with the given sample name from the database.

    This function queries the 'samples' table in the database for a row where the 'name'
    column matches the provided 'name' argument. If a match is found, it returns the
    corresponding 'sample_id'.

    :param name: str - The name of the sample to search for.

    :return: int - The sample_id associated with the given sample name.
    """
    conn = sqlite3.connect(MEZO_DB)
    cur = conn.cursor()
    try:
        cur.execute(
            'SELECT sample_id FROM samples WHERE name = ?', (name,)
        )
        sample_id = cur.fetchone()[0]
        return sample_id
    except sqlite3.Error as e:
        print(f"Error while getting sample_id from name: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def open_editor(e: ft.ControlEvent, page: ft.Page) -> None:
    """
    Open the editor page for a specific sample.

    This function takes an event and a page as input, retrieves the sample name
    from the event, and finds the corresponding sample_id from the database. If
    the sample_id exists, it sets the 'sample_id' and 'image_index' in the session
    and navigates to the editor page.

    :param e: flet.ControlEvent - The event containing control details.
    :param page: flet.Page - The current page object.

    :return: None
    """
    name = e.control.content.controls[1].value
    sample_id = get_id_from_name(name)
    if sample_id is None:
        return
    page.session.set('sample_id', sample_id)
    page.session.set('image_index', 0)
    page.go('/editor')


def init_samples_library(page: ft.Page) -> None:
    """
    Initialize the samples library.

    This function retrieves the overview of all samples from the database and
    creates a ft.GridView with a card for each sample. The card contains a preview
    of the first image in the sample, the sample name, and the creation date of the
    sample. The cards are clickable and open the sample in the editor when clicked.

    If the samples library is empty, it sets the 'samples_library' in the session
    to a text prompting the user to add a sample.

    :param page: flet.Page - The current page object.

    :return: None
    """
    # Get samples overview
    samples = get_samples_overview()

    # Display a message if the library is empty
    if len(samples) == 0:
        page.session.set(
            'samples_library',
            ft.Text('Чтобы добавить образец, нажмите кнопку "Новый образец" -->')
        )
        return

    # Create library grid
    samples_library = ft.GridView(
        runs_count=5,
        max_extent=150,
        child_aspect_ratio=0.75,
        spacing=5,
        run_spacing=5,
        expand=True,
    )

    # Create card for each sample
    for sample in samples:
        name = sample[2]
        date = datetime.strptime(sample[1], "%Y-%m-%d %H:%M:%S.%f").strftime('%d.%m.%Y %H:%M')

        # Set preview image
        image_preview_path = os.path.join(
            DATA_DIR, sample[2], 'low', f"img {sample[5] + 1}.jpg"
        )
        result_preview_path = os.path.join(
            DATA_DIR, sample[2], 'result', f"img {sample[5] + 1}.jpg"
        )
        preview = result_preview_path if os.path.exists(result_preview_path) else image_preview_path

        # Add card to library
        samples_library.controls.append(
            ft.Container(
                ft.Column(
                    [
                        ft.Stack(
                            [
                                ft.Image(
                                    src=preview,
                                    fit=ft.ImageFit.COVER,
                                    repeat=ft.ImageRepeat.NO_REPEAT,
                                    border_radius=ft.border_radius.all(10),
                                ),
                                ft.Container(
                                    ft.Row(
                                        [
                                            ft.Icon(
                                                ft.Icons.PHOTO_LIBRARY,
                                                size=14,
                                                color=COLOR_SCHEME['background']
                                            ),
                                            ft.Text(
                                                str(sample[4]),
                                                color=COLOR_SCHEME['background']
                                            )
                                        ],
                                        alignment=ft.MainAxisAlignment.CENTER,
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                        width=40,
                                        spacing=5
                                    ),
                                    bgcolor=COLOR_SCHEME['highlight'],
                                    border_radius=ft.border_radius.all(10),
                                    padding=ft.padding.all(1),
                                )
                            ],
                            alignment=ft.Alignment(0.9, 0.9)
                        ),
                        ft.Text(name),
                        ft.Text(date, size=12, color='grey')
                    ],
                    spacing=3
                ),
                ink=True,
                on_click=lambda e: open_editor(e, page),
            )
        )

    # Set samples library to session storage
    page.session.set('samples_library', samples_library)


def library(page: ft.Page) -> ft.View:
    """
    Initialize the samples library view.

    This function creates a view with all the controls for the samples library.
    Library view contains a button to create a new sample and the samples library.

    :param page: flet.Page - The current page object.

    :return: flet.View - The view with all the controls.
    """
    def add_sample_to_db(name: str, description: str, count: int) -> None:
        """
        Add a new sample to the database.

        This function adds a new sample to the 'samples' table in the database with the given
        name, description and count. It also adds the appropriate number of rows to the 'images'
        table, each with the sample_id of the newly added sample.

        :param name: str - The name of the sample to add.
        :param description: str - The description of the sample to add.
        :param count: int - The number of images in the sample to add.

        :return: None
        """
        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO samples (date, name, description, count) VALUES (?, ?, ?, ?)",
                (datetime.now(), name, description, count)
            )
            cur.execute(
                "SELECT sample_id FROM samples ORDER BY sample_id DESC LIMIT 1")
            last_id = cur.fetchone()[0]
            for _ in range(count):
                cur.execute(
                    "INSERT INTO images (sample_id) VALUES (?)",
                    (last_id,)
                )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error while adding new sample to the database: {e}")
        finally:
            cur.close()
            conn.close()

    def name_is_unique(name: str) -> bool:
        """
        Check if a sample name is unique in the database.

        :param name: str - The name of the sample to check for uniqueness.

        :return: True if the sample name is unique, False otherwise.
        """
        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            cur.execute("SELECT * FROM samples WHERE name = ?", (name,))
            sample = cur.fetchone()
            return sample is None
        except sqlite3.Error as e:
            print(f"Error while checking sample name uniqueness: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def new_sample(e: ft.ControlEvent) -> None:
        """
        Handle the creation of a new sample.

        This function is called when the "Create" button is clicked in the new sample dialog.
        It checks that the sample name is not empty and unique. If the sample name is valid and
        images is added, it creates a new sample in the database, creates the appropriate folders
        for the sample, and saves the sample images to the 'low' folder. It then updates the samples
        library and closes the new sample dialog.

        :param e: flet.ControlEvent - The event that triggered this function.

        :return: None
        """
        if e is None:
            return

        # Checking sample name uniqueness and validaty
        name = new_sample_dialog.content.controls[2].value
        name_field_error = new_sample_dialog.content.controls[2].error_text
        if name is None or name == '':
            new_sample_dialog.content.controls[2].error_text = 'Обязательное поле'
            new_sample_dialog.content.controls[2].update()
            return
        if name_field_error:
            return

        # Checking if images is added
        if isinstance(new_sample_dialog.content.controls[0].content, ft.Text):
            new_sample_dialog.content.controls[0].border = ft.border.all(1, '#FFB4AB')
            new_sample_dialog.content.controls[0].content.style = ft.TextStyle(color='#FFB4AB')
            new_sample_dialog.content.controls[0].update()
            return

        description = new_sample_dialog.content.controls[3].value
        images = new_sample_dialog.content.controls[0].content.controls
        count = len(images)

        # Update new sample dialog
        new_sample_dialog.content.controls = [
            ft.Container(
                content=ft.Column(
                    [
                        ft.ProgressRing(color=COLOR_SCHEME['highlight']),
                        ft.Text('Создание нового образца...', size=14, color='grey')
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER
                ),
                alignment=ft.alignment.center
            )
        ]
        new_sample_dialog.update()

        # Creation sample directories
        os.makedirs(os.path.join(DATA_DIR, name, 'low'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, name, 'result'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, name, 'masks'), exist_ok=True)

        # Saving low-weight .jpg sample images in `low` folder
        try:
            for i, file in enumerate(images):
                image = Image.open(file.src)
                image.save(os.path.join(DATA_DIR, name, 'low', f"img {i + 1}.jpg"))
                image.close()
        except (IOError, OSError, TypeError, AttributeError) as error:
            print(f"Error while saving image: {error}")
            page.close(new_sample_dialog)
            return

        # Adding sample to the database
        add_sample_to_db(name, description, count)

        # Updating samples library
        init_samples_library(page)
        view.controls[0].controls[0].controls[0].content = page.session.get('samples_library')
        view.controls[0].controls[0].controls[0].update()

        page.close(new_sample_dialog)

    def open_file(e: ft.FilePickerResultEvent) -> None:
        """
        Open the file picker dialog and displays the selected files in a grid.

        This function is called when the user try to add sample images.
        If the user selects any files, it creates a grid with a preview of each file
        and updates the content of the new sample dialog with this grid.

        :param e: flet.FilePickerResultEvent - The event containing the list of selected files.

        :return: None
        """
        if e.files is None:
            return

        # Initializing preview grid
        preview = ft.GridView(
            auto_scroll=True,
            child_aspect_ratio=0.8,
            horizontal=True,
            expand=True,
        )

        # Adding images to the preview grid
        for file in e.files:
            preview.controls.append(
                ft.Image(
                    src=file.path,
                    fit=ft.ImageFit.CONTAIN,
                )
            )

        # Updating new sample dialog
        new_sample_dialog.content.controls[0].content = preview
        new_sample_dialog.content.controls[0].border = None
        new_sample_dialog.content.controls[1].value = \
            f"{len(e.files)} фото. Для прокрутки - зажмите shift."
        new_sample_dialog.update()

    def name_field(e: ft.ControlEvent) -> None:
        """
        Handle the sample name field on change event.

        This function is called when the sample name text field changes.
        It checks that the sample name is unique and updates the error text
        of the sample name text field accordingly.

        :param e: flet.ControlEvent - The event containing the new sample name.

        :return: None
        """
        new_sample_dialog.content.controls[2].error_text = None
        if not name_is_unique(e.data):
            new_sample_dialog.content.controls[2].error_text = 'Такое имя уже существует'
        new_sample_dialog.content.controls[2].update()

    def create_new_sample_dialog() -> ft.AlertDialog:
        """
        Create a new sample dialog.

        This function returns a ft.AlertDialog with a form for creating a new sample.
        The form consists of a container for uploading images, a text field for the sample name,
        a text area for the sample description, and a button for creating the sample.

        :return: ft.AlertDialog - The new sample dialog.
        """
        return ft.AlertDialog(
            title=ft.Text('Новый образец', text_align=ft.TextAlign.CENTER),
            bgcolor=COLOR_SCHEME['background'],
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text('Загрузить фотографии'),
                        border=ft.border.all(1, ft.Colors.BLACK),
                        border_radius=25,
                        alignment=ft.alignment.center,
                        height=200,
                        ink=True,
                        on_click=lambda _: fp.pick_files(
                            file_type=ft.FilePickerFileType.IMAGE, allow_multiple=True)
                    ),
                    ft.Text('', size=11, color='grey'),
                    ft.TextField(
                        label='Название',
                        on_change=name_field
                    ),
                    ft.TextField(
                        label='Описание',
                        multiline=True,
                        min_lines=2,
                        max_lines=3
                    ),
                    ft.Container(
                        content=ft.ElevatedButton(
                            'Создать образец',
                            style=ft.ButtonStyle(
                                color=COLOR_SCHEME['background'],
                                bgcolor=COLOR_SCHEME['highlight'],
                                overlay_color='#B098F9',
                            ),
                            on_click=new_sample
                        ),
                        expand=True,
                        alignment=ft.alignment.bottom_center
                    )

                ],
                width=400,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            ),
        )

    def open_new_sample_dialog(e: ft.ControlEvent) -> None:
        """
        Open the new sample dialog.

        :param e: flet.ControlEvent - The event that triggered this function.

        :return: None
        """
        global new_sample_dialog

        if e is None:
            return
        new_sample_dialog = create_new_sample_dialog()
        page.open(new_sample_dialog)

    # Initializing file picker
    fp = ft.FilePicker(on_result=open_file)
    page.overlay.append(fp)

    # Initializing library view
    view = ft.View(
        '/library',
        controls=[
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.Container(
                                page.session.get('samples_library'),
                                margin=ft.margin.all(5),
                                alignment=ft.alignment.center,
                                expand=True
                            ),
                        ],
                        expand=True
                    ),
                    ft.VerticalDivider(width=1),
                    ft.Column(
                        [
                            ft.Container(
                                ft.IconButton(
                                    icon=ft.Icons.CLOSE,
                                    on_click=lambda _: page.window.destroy(),
                                    icon_color=COLOR_SCHEME['highlight']
                                ),
                                alignment=ft.alignment.top_right
                            ),
                            ft.Column(
                                [
                                    ft.Image(src=SVG, width=200),
                                    ft.Text('Mezo', size=24),
                                    ft.Text('AI-powered mesophase analysis', size=12, color='grey')
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER
                            ),
                            ft.Container(
                                ft.ElevatedButton(
                                    'Новый образец',
                                    style=ft.ButtonStyle(
                                        color=COLOR_SCHEME['background'],
                                        bgcolor=COLOR_SCHEME['highlight'],
                                        overlay_color='#B098F9',
                                    ),
                                    on_click=open_new_sample_dialog
                                ),
                            ),
                            ft.Column(
                                [
                                    ft.Text('Author:', size=12, color='grey'),
                                    ft.TextButton(
                                        'Roman Kozlov',
                                        style=ft.ButtonStyle(
                                            text_style=ft.TextStyle(size=12),
                                            color=COLOR_SCHEME['highlight']),
                                        on_click=lambda _: page.launch_url(
                                            'https://project11648075.tilda.ws/')
                                    )
                                ],
                                spacing=5,
                                alignment=ft.MainAxisAlignment.END,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER
                            )

                        ],
                        width=250,
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER
                    )
                ],
                expand=True
            )
        ],
        bgcolor=COLOR_SCHEME['background']
    )

    return view
