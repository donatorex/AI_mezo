"""
report.py - Mezo module script for build HTML report in the Mezo application.

This script provides functions for generating reports based on sample data stored in a SQLite
database. It includes functionalities to retrieve sample data, fill HTML templates with the data,
and generate HTML reports displaying annotated images of the samples.

Key Functions:
- get_report_data: Retrieves sample data from the database for a given sample ID, including details
                   about images and mezo data associated with the sample.

- report_filling: Replaces a specified placeholder in the HTML template with a provided value.

- get_report: Generates an HTML report by filling placeholders in a template using values from
              a given dictionary.

- generate_html: Creates an HTML string for a report that arranges original and annotated images in
                 a gallery format, with pagination.

- create_pdf_report: Generates a PDF report for a specified sample ID, retrieves the relevant sample
                     data, and writes an HTML report to a file which is then opened in the default
                     web browser.

Constants:
- DATA_DIR: str - The directory where sample data images are stored.
- MEZO_DB: str - The name of the SQLite database containing sample data.

Author: Roman Kozlov
Github: https://github.com/donatorex
"""

import os
import sqlite3
import webbrowser
from datetime import datetime
from typing import Union

import numpy as np
from PIL import Image

DATA_DIR = 'data'
MEZO_DB = 'mezo.db'


def get_report_data(sample_id: int) -> Union[dict, None]:
    """
    Get sample data from database by sample_id.

    The function takes a sample_id integer as input, and retrieves the corresponding
    sample data from the database. The sample data is stored in a dictionary with the
    following keys:

    - 'sample_id': int - The id of sample.
    - 'date': str - The date of sample.
    - 'name': str - The name of sample.
    - 'description': str - The description of sample.
    - 'count': int - The count of images in sample.
    - 'preview': str - The path to the preview image of sample.
    - 'images': dict - A dictionary of images in sample. The keys of the dictionary are
      the index of image in sample, and the values are dictionaries with the following
      keys:

      - 'image_id': int - The id of image.
      - 'porosity': float - The porosity of image.
      - 'scale_px': float - The scale of image in pixels.
      - 'scale_mkm': float - The scale of image in micrometers per kilometer.
      - 'mezo': dict - A dictionary of mezo in image. The keys of the dictionary are the
        index of mezo in image, and the values are dictionaries with the following keys:

        - 'mezo_id': int - The id of mezo.
        - 'center': tuple - The center of mezo (x, y).
        - 'diameter': float - The diameter of mezo.
        - 'square': float - The square of mezo.

    If the retrieval is successful, the function returns the sample data as a dictionary.
    Otherwise, it returns None.

    :param sample_id: int - The id of sample to get data from.

    :return: dict or None
    """
    conn = sqlite3.connect(MEZO_DB)
    cur = conn.cursor()
    try:
        # Get sample data
        cur.execute(
            'SELECT * FROM samples WHERE sample_id = ?',
            (sample_id,)
        )
        samples_rows = cur.fetchone()

        # Get images data
        cur.execute(
            'SELECT * FROM images WHERE sample_id = ?',
            (sample_id,)
        )
        images_rows = cur.fetchall()
        image_ids = [row[0] for row in images_rows]

        # Get mezo data
        cur.execute(
            f"SELECT * FROM mezo_data WHERE image_id IN ({', '.join('?' * len(image_ids))})",
            image_ids
        )
        mezo_rows = cur.fetchall()

        # Build a dict
        sample_data = {
            'sample_id': samples_rows[0],
            'date': samples_rows[1],
            'name': samples_rows[2],
            'description': samples_rows[3],
            'count': samples_rows[4],
            'preview': samples_rows[5],
            'images': {
                i: {
                    'image_id': image_row[0],
                    'porosity': image_row[2],
                    'scale_px': image_row[3],
                    'scale_mkm': image_row[4],
                    'mezo': {
                        i: {
                            'mezo_id': mezo_row[0],
                            'center': (mezo_row[2], mezo_row[3]),
                            'diameter': mezo_row[4],
                            'square': mezo_row[5]
                        } for i, mezo_row in enumerate(mezo_rows) if mezo_row[1] == image_row[0]
                    }
                } for i, image_row in enumerate(images_rows)
            }
        }
        return sample_data
    except sqlite3.Error as e:
        print(f"Error while getting report data: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def report_filling(template: str, placeholder: str, value: str) -> str:
    """
    Replace a placeholder with a value in an HTML template.

    :param template: str - The HTML template as a string.
    :param placeholder: str - The placeholder string to replace.
    :param value: str - The value to replace the placeholder with.

    :return: str - The modified HTML template as a string.
    """
    return template.replace(placeholder, value)


def get_report(params: dict) -> str:
    """
    Generate a report by filling placeholders in an HTML template.

    This function reads an HTML template from a file and replaces placeholders
    with corresponding values provided in the `params` dictionary. Each key in
    the `params` dictionary corresponds to a placeholder in the template, and
    each value is the replacement text.

    :param params: dict - A dictionary where keys are placeholders in the
                        template and values are the text to replace them with.

    :return: str - The filled HTML template as a string.
    """
    with open("report template/index.html", "r", encoding="utf-8") as file:
        template = file.read()

    for placeholder, value in params.items():
        template = report_filling(template, placeholder, value)

    return template


def generate_html(images: list) -> str:
    """
    Generate HTML code for a report with annotated images.

    This function takes a list of image pairs (original image, annotated image)
    and generates an HTML string with the images arranged in a gallery. The
    gallery is divided into pages of 3 images each, with the original and
    annotated images side-by-side. The function returns the HTML string.

    :param images: list - A list of tuples, where each tuple contains the
                          original image and the annotated image as strings.
    :return: str - The generated HTML string.
    """
    html = ''
    page_number = 2

    # Add pair of original and annotated images
    for index, image_pair in enumerate(images):
        original_image, annotated_image = image_pair
        caption_number = index + 1

        # Add gallery
        if index % 3 == 0:
            html += '    <article>\n'
            html += '        <div class="gallery">\n'

        # Add gallery item
        html += '            <div class="gallery-item">\n'

        # Add title in the head of the page
        if index % 3 == 0:
            html += '                <p>Исходные изображения</p>\n'

        # Add original image
        html += f'                <img src="{original_image}">\n'
        html += f'                <div class="caption">Снимок {caption_number}</div>\n'
        html += '            </div>\n'
        html += '            <div class="gallery-item">\n'

        # Add title in the head of the page
        if index % 3 == 0:
            html += '                <p>Аннотированные изображения</p>\n'

        # Add annotated image
        html += f'                <img src="{annotated_image}">\n'
        html += f'                <div class="caption">Снимок {caption_number}</div>\n'
        html += '            </div>\n'

        # Add number page in the bottom of the page
        if (index - 2) % 3 == 0 and index != len(images) - 1:
            html += '        </div>\n'
            html += f'        <div class="page">Страница {page_number}</div>\n'
            html += '    </article>\n'

            page_number += 1

    # Add number page in the bottom of the page
    html += '        </div>\n'
    html += f'        <div class="page">Страница {page_number}</div>\n'
    html += '    </article>\n'

    return html


def create_pdf_report(sample_id: int) -> None:
    """
    Generate a PDF report with annotated images.

    This function takes a sample_id, retrieves the sample data from the database,
    and generates an HTML report with the images arranged in a gallery. The
    gallery is divided into pages of 3 images each, with the original and
    annotated images side-by-side. The function then writes the HTML to a file
    and opens the file in the default browser.

    :param sample_id: int - The ID of the sample to generate the report for.
    """
    # Get sample data
    sample_data = get_report_data(sample_id)

    # Sample data
    name = sample_data['name']
    date = datetime.strptime(sample_data['date'], "%Y-%m-%d %H:%M:%S.%f").strftime('%d.%m.%Y')
    description = sample_data['description'] if sample_data['description'] else 'Не указано'
    count = sample_data['count']

    # Initialize lists for tables and images gallery
    size_rows = [[] for _ in range(7)]
    results_rows = [[] for _ in range(5)]
    images_rows = []

    # Add sample images
    for i, image in enumerate(sample_data['images'].values()):
        # Get image data
        scale_px = image['scale_px']
        scale_mkm = image['scale_mkm']
        scale_factor = scale_mkm / scale_px
        porosity = image['porosity'] * 100

        # Append original and annotated images (if any mezo is exists) in the gallery list
        image_path = os.path.join(DATA_DIR, name, 'low', f"img {i + 1}.jpg")
        result_path = os.path.join(DATA_DIR, name, 'result', f"img {i + 1}.jpg")
        if not os.path.exists(result_path):
            result_path = image_path
        image_file = Image.open(image_path)
        image_width, image_height = image_file.size
        image_file.close()

        images_rows.append(
            (
                os.path.join(os.getcwd(), image_path),
                os.path.join(os.getcwd(), result_path)
            )
        )

        # Get mezo sizes and squares
        mezo_size_list = [d['diameter'] * scale_factor for d in image['mezo'].values()]
        mezo_square_list = [d['square'] * scale_factor ** 2 for d in image['mezo'].values()]

        # Size distribution
        size_rows[0].append(sum(map(lambda x: x <= 2, mezo_size_list)))
        size_rows[1].append(sum(map(lambda x: 2 < x <= 5, mezo_size_list))),
        size_rows[2].append(sum(map(lambda x: 5 < x <= 10, mezo_size_list))),
        size_rows[3].append(sum(map(lambda x: 10 < x <= 20, mezo_size_list))),
        size_rows[4].append(sum(map(lambda x: 20 < x <= 50, mezo_size_list))),
        size_rows[5].append(sum(map(lambda x: x > 50, mezo_size_list))),
        size_rows[6].append(len(mezo_size_list))

        # Analytical results
        max_size = max(mezo_size_list, default=0)
        material_square = 0.000001 * (1 - 0.01 * porosity) * \
            image_width * image_height * scale_factor ** 2
        mezo_square = 0.000001 * sum(mezo_square_list)
        mezo_percentage = 100 * mezo_square / material_square

        results_rows[0].append(max_size)
        results_rows[1].append(porosity)
        results_rows[2].append(material_square)
        results_rows[3].append(mezo_square)
        results_rows[4].append(mezo_percentage)

    # Build params dict for HTML report
    params = {
        '{{name}}': name,
        '{{date}}': date,
        '{{description}}': description,
        '{{count}}': str(count),
        '{{table_headers}}': ''.join(f"<th>{i + 1}</th>" for i in range(count))
    }

    # Fill the size distribution table
    for i, row in enumerate(size_rows):
        params[f'{{{{size_row_{i + 1}}}}}'] = \
            ''.join(f"<td>{value}</td>" for value in row) + f"<td>{sum(row)}</td>"

    # Fill the analytical results table
    for i, row in enumerate(results_rows):
        if i == 0:
            tr = ''.join(f"<td>{value:.2f}</td>" for value in row) + f"<td>{max(row):.2f}</td>"
        elif i == 1 or i == 4:
            tr = ''.join(f"<td>{value:.3f}</td>" for value in row) + f"<td>{np.mean(row):.3f}</td>"
        else:
            tr = ''.join(f"<td>{value:.3f}</td>" for value in row) + f"<td>{sum(row):.3f}</td>"
        params[f'{{{{results_row_{i + 1}}}}}'] = tr

    # Fill the images gallery
    image_pages = generate_html(images_rows)
    params['{{image_pages}}'] = image_pages

    # Generate HTML report
    html_content = get_report(params)

    # Save HTML report
    with open(os.path.join("report template", "last_report.html"), "w", encoding="utf-8") as file:
        file.write(html_content)

    # Open HTML report in browser
    webbrowser.open(
        os.path.join('file://', os.getcwd(), os.path.join("report template", "last_report.html"))
    )
