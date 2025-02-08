"""
Mezo - AI-Powered Mesophase Analysis Application.

Mezo designed for managing and analyzing samples of pitch and their associated images. It provides a
user-friendly interface for viewing a library of samples and editing images.

Features:
- Database Management: The application uses SQLite to manage a local database (mezo.db) that stores
                       information about samples, images, and their associated analysis data.
- Sample Library: Users can view and manage a library of samples, including their descriptions and
                  associated images.
- Image Editor: The application includes an image editing interface that allows users to analyze
                and segmenate images related to the samples.

Main Functions:
- `create_database()`: Creates the necessary database schema with three tables: samples, images
                       and mezo_data.
- `main(page: ft.Page)`: The entry point of the application that initializes the main window and
                         handles routing between different views (library and editor).
- `start()`: Sets up the initial loading view of the application with a spinner and navigates to
             the sample library after a brief delay.
- `route_change(e: ft.RouteChangeEvent)`: Handles route changes and updates the page view
                                          accordingly based on the selected route.

Constants:
- `COLOR_SCHEME`: A dictionary defining the color scheme used throughout the application.
- `SVG`: An SVG string that represents a visual element in the application interface.
- `DATA_DIR`: The directory where data files are stored.
- `MEZO_DB`: The name of the SQLite database file.

Usage:
To run the application, execute the script, which will create the database if it does not exist
and launch the Flet application interface.

Dependencies:
- Flet: A Python library for building interactive web applications.
- SQLite3: A lightweight database engine for managing application data.

Author: Roman Kozlov
Github: https://github.com/donatorex
"""

import os
import sqlite3
import time

import flet as ft
from mezo import editor, init_samples_library, library

COLOR_SCHEME = {
    'primary': '#C3C7CF',
    'secondary': '#535353',
    'highlight': '#835AFF',
    'background': '#282828'
}

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
  <title>Слой 1</title>Б
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

DATA_DIR = 'data'
MEZO_DB = 'mezo.db'


def create_database() -> None:
    """
    Create the database schema in the mezo.db file.

    This function creates three tables: samples, images, and mezo_data. The samples table stores
    information about the samples, such as name, description, and count of images. The images table
    stores information about the images, such as porosity and scale. The mezo_data table stores
    the data for each mezo, such as center_x, center_y, diameter, and square.

    The function first connects to the mezo.db file, then tries to execute the SQL commands to
    create the tables. If an error occurs, it prints the error message. Finally, it closes
    the connection to the database.

    :return: None
    """
    conn = sqlite3.connect('mezo.db')
    cur = conn.cursor()
    try:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                count INTEGER NOT NULL,
                preview INTEGER DEFAULT 0
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER,
                porosity REAL DEFAULT 0,
                scale_px INTEGER DEFAULT 1,
                scale_mkm INTEGER DEFAULT 1,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS mezo_data (
                mezo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                center_x REAL DEFAULT 0,
                center_y REAL DEFAULT 0,
                diameter REAL DEFAULT 0,
                square REAL DEFAULT 0,
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            )
        ''')
    except sqlite3.Error as e:
        print(f"Error while creating database: {e}")
    finally:
        cur.close()
        conn.close()


def main(page: ft.Page) -> None:
    """
    Mezo application entry point.

    This function is called when the application starts. It sets up the main window
    and starts the application by calling the `start` function.

    The `start` function adds a view to the page with a spinner and a title, and then
    waits for 3 seconds before navigating to the '/library' route.

    The `route_change` function is called when the page route changes. It clears the
    current view and adds a new view based on the route.

    The '/library' route shows the samples library, and the '/editor' route shows the
    image editor.

    If the 'mezo.db' file does not exist, the `create_database` function is called to
    create it. This function creates the database tables for samples, images, and
    mezo data.

    :param page: flet.Page - The Flet page object.
    :return: None
    """
    # Disable the visibility of the window so as not to show the effect of them resizing
    page.window.visible = False
    page.update()
    time.sleep(1)

    # Initialize the main window
    page.title = 'Mezo'
    page.window.icon = os.path.join(os.getcwd(), 'Mezo.ico')
    page.window.bgcolor = COLOR_SCHEME['background']
    page.window.title_bar_hidden = True
    page.window.resizable = False
    page.window.width = 300
    page.window.height = 400
    page.window.center()

    # Show the window
    page.window.visible = True
    page.update()

    def start() -> None:
        """
        Initialize the start view of the Mezo application.

        This function sets up the initial view of the application with a spinner and title.
        It creates a new view with a loading screen displaying the application name, a description,
        and a progress ring. If the 'samples_library' is not present in the session, it initializes
        the samples library. After a brief delay, it navigates to the '/library' route.

        :return: None
        """
        # Add start view
        page.views.append(
            ft.View(
                '/start',
                controls=[
                    ft.Container(
                        ft.Column(
                            [
                                ft.Image(src=SVG, width=200),
                                ft.Text('Mezo', size=24),
                                ft.Text('AI-powered mesophase analysis', size=12, color='grey'),
                                ft.Column(
                                    [
                                        ft.ProgressRing(color=COLOR_SCHEME['highlight'])
                                    ],
                                    height=50,
                                    alignment=ft.MainAxisAlignment.END
                                )
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER
                        ),
                        alignment=ft.alignment.center,
                        expand=True
                    )
                ],
                bgcolor=COLOR_SCHEME['background']
            )
        )
        page.update()

        # Initialize samples library
        if not page.session.contains_key('samples_library'):
            init_samples_library(page)

        # Delay
        time.sleep(3)

        # Open samples library
        page.go('/library')

    def route_change(e: ft.RouteChangeEvent) -> None:
        """
        Route change event handler.

        Handles route changes and updates the page accordingly. Currently
        supported routes are '/library' and '/editor'.

        :param e: flet.RouteChangeEvent object
        :return: None
        """
        # Disable the visibility of the window so as not to show the effect of them resizing
        page.window.visible = False
        page.update()

        # Clear the current view
        page.views.clear()
        time.sleep(0.2)

        if e.route == '/library':
            # Change the window settings
            page.window.title_bar_hidden = True
            page.window.resizable = False
            page.window.width = 1000
            page.window.height = 600
            page.window.center()
            time.sleep(1)

            # Initialize samples library
            init_samples_library(page)

            # Add library view
            page.views.append(
                library(page)
            )

            # Show the window
            page.window.visible = True

        elif e.route == '/editor':
            # Change the window settings
            page.window.title_bar_hidden = False
            page.window.resizable = True
            page.window.min_width = 1000
            page.window.min_height = 600
            page.window.width = 1200
            page.window.height = 800
            page.window.center()
            time.sleep(1)

            # Add editor view
            page.views.append(
                editor(page)
            )

            # Show the window
            page.window.visible = True

        page.update()

    # Set the route change handler
    page.on_route_change = route_change

    # Create the database if it doesn't exist
    if not os.path.exists('mezo.db'):
        create_database()

    # Initialize the start page view
    start()


if __name__ == '__main__':
    ft.app(target=main)
