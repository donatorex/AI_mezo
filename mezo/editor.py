"""
editor.py - Mezo module script for editing sample image in the Mezo application.

This script provides the functionality to create and manage the editor view of the Mezo application.
It allows users to interact with images and mezo data, providing tools for selecting, adding, and
removing mezophases from images. The editor supports functionalities such as zooming, panning,
and managing the viewer's state. It also integrates with a database to store and retrieve mezo
data and image information.

Key functionalities include:
- Loading and displaying images with associated masks.
- Tools for selecting mezophases:
        Magic Select: Select mezophases using a AI model (Segment Anything by Meta);
        Manual Select: Select mezophases manually;
        Remove Tool: Remove selected mezophases.
- Interaction handling for zooming and moving the viewer.
- Saving and updating mezo data in the database.
- Updating the user interface components such as status bars, menus, and dialogs.
- Handling user inputs through keyboard events and control interactions.

The main function is `editor(page: ft.Page)`, which initializes the editor view and sets up the
necessary controls and event handlers.

Functions:
- editor(page): Initializes the editor view with controls for image editing.
- get_sample_data(sample_id): Retrieves sample data from the database.
- get_mezo_data(image_id): Retrieves mezo data from the database.
- update_control(control): Updates the specified control if the page has views.
- update_status_bar(): Updates the status bar with viewer and image scale information.
- update_status_message(message): Updates the status message in the status bar.
- update_scale(scale): Updates the viewer and image scale based on the provided scale factor.
- window_resized(e): Handles window resize events and updates viewer size.
- viewer_interaction_start(e): Handles the start of viewer interactions.
- viewer_interaction_update(e): Handles updates during viewer interactions.
- viewer_interaction_end(e): Handles the end of viewer interactions.
- add_mezo_to_db(center, diameter): Adds a mezo to the database.
- add_mezo_to_viewer(x, y, radius, recursive): Adds a mezo visualization to the viewer.
- magic_select(coord): Uses the Magic Select tool to select a mezo.
- manual_select(coord): Uses the Manual Select tool to add a mezo.
- remove_tool(coord): Removes a mezo based on user clicks.
- open_image(sample_id, image_index): Opens an image and initializes viewer settings.
- save_image(): Saves the current image and mask as a result image.
- arrow_styling(image_index, image_count): Changes the style of arrow buttons in the menubar.
- previous_image(e): Handles the event of clicking the previous image button.
- next_image(e): Handles the event of clicking the next image button.
- update_image_data(image_index, parameter, value): Updates image data in the database.
- open_results_table(e): Opens the results table with data on the current image.
- change_porosity(e): Changes the porosity of the material.
- change_scale_factor(e): Changes the scale factor of the image.
- sam_init(): Initializes the SAM model.
- change_tool(e): Changes the current tool based on user selection.
- reset_viewer(e): Resets the viewer to the default state.
- on_keyboard(e): Handles keyboard events for various actions.
- remove_mezo(key): Removes a mezo from the mask and database.
- delete_sample(): Deletes a sample from the database and file system.
- to_library(e): Navigates to the library page.

Constants:
- COLOR_SCHEME: A dictionary defining color codes for various UI elements.
- DATA_DIR: Directory path for storing sample data.
- MEZO_DB: Database filename for storing sample information.

Author: Roman Kozlov
Github: https://github.com/donatorex
"""

import base64
import io
import math
import os
import shutil
import sqlite3
from typing import Union

import cv2
import flet as ft
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
from segment_anything import SamPredictor, sam_model_registry

from .report import create_pdf_report

COLOR_SCHEME = {
    'primary': '#C3C7CF',
    'secondary': '#535353',
    'highlight': '#835AFF',
    'background': '#282828'
}

DATA_DIR = 'data'
MEZO_DB = 'mezo.db'

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"


def editor(page: ft.Page) -> ft.View:
    """
    Initialize the editor view.

    This function creates a view with all the controls for the editor.
    Editor view contains a canvas with stack of sample image and mask, statusbar, menubar and
    toolbar with next tools: Manual Select, Magic Select and Remove tool.

    :param page: flet.Page - The current page object.

    :return: flet.View - The view with all the controls.
    """
    def get_sample_data(sample_id: int) -> None:
        """
        Get sample data from database by sample_id.

        This function gets sample data from database by sample_id and puts it into the session
        storage.

        :param sample_id: int - The id of sample to get data from.

        :return: None
        """
        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            cur.execute(
                'SELECT * FROM samples WHERE sample_id = ?',
                (sample_id,)
            )
            samples_rows = cur.fetchone()
            cur.execute(
                'SELECT * FROM images WHERE sample_id = ?',
                (sample_id,)
            )
            images_rows = cur.fetchall()

            sample_data = {
                'sample_id': samples_rows[0],
                'date': samples_rows[1],
                'name': samples_rows[2],
                'description': samples_rows[3],
                'count': samples_rows[4],
                'preview': samples_rows[5],
                'images': {
                    i: {
                        'image_id': row[0],
                        'porosity': row[2],
                        'scale_px': row[3],
                        'scale_mkm': row[4]
                    } for i, row in enumerate(images_rows)
                }
            }
            page.session.set('sample_data', sample_data)
        except sqlite3.Error as e:
            print(f"Error while getting sample data: {e}")
        finally:
            cur.close()
            conn.close()

    def get_mezo_data(image_id: int) -> None:
        """
        Get mezo data from the database by image_id.

        This function queries the mezo_data table in the database for entries associated
        with the specified image_id. If mezo data is found, it is formatted into a dictionary
        and stored in the session. If no data is found, an empty dictionary is stored in the
        session.

        :param image_id: int - The ID of the image to retrieve mezo data for.

        :return: None
        """
        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            cur.execute(
                'SELECT * FROM mezo_data WHERE image_id = ?', (image_id,)
            )
            mezo_rows = cur.fetchall()
            if mezo_rows:
                mezo_data = {
                    i: {
                        'mezo_id': row[0],
                        'image_id': row[1],
                        'center': (row[2], row[3]),
                        'diameter': row[4],
                        'square': row[5]
                    } for i, row in enumerate(mezo_rows)
                }
                page.session.set('mezo_data', mezo_data)
            else:
                page.session.set('mezo_data', {})
        except sqlite3.Error as e:
            print(f"Error while getting mezo data: {e}")
        finally:
            cur.close()
            conn.close()

    def update_control(control: ft.Control) -> None:
        """
        Update the control if the page has at least one view.

        This function helps to update the state of a control if the view already exists.

        :param control: flet.Control - The control to update.

        :return: None
        """
        if len(page.views) > 0:
            control.update()

    def update_status_bar() -> None:
        """
        Update the status bar with current viewer and image scale information.

        This function retrieves various status information from the session and updates
        the status bar controls accordingly. The viewer scale, image scale, and viewer
        offset are displayed if available in the session. Additionally, pixel-to-micron
        scale information is shown if sample data exists in the session.

        :return: None
        """
        if page.session.contains_key('viewer_scale'):
            status_bar.content.controls[0].controls[0].value = \
                f"viewer scale: {page.session.get('viewer_scale'):.2f}x;"
        if page.session.contains_key('image_scale'):
            status_bar.content.controls[0].controls[1].value = \
                f"image scale: {page.session.get('image_scale'):.2f}x;"
        if page.session.contains_key('viewer_offset'):
            viewer_offset = page.session.get('viewer_offset')
            status_bar.content.controls[0].controls[2].value = \
                f"offset: ({viewer_offset.x}, {viewer_offset.y})"
        if page.session.contains_key('sample_data'):
            image_data = page.session.get('sample_data')['images'][page.session.get('image_index')]
            scale_px = image_data['scale_px']
            scale_mkm = image_data['scale_mkm']

            status_bar.content.controls[2].controls[2].value = f"{scale_px} px = {scale_mkm} µm"

        update_control(status_bar)

    def update_status_message(message: str) -> None:
        """
        Update the status bar with the given message.

        This function updates the message placeholder in the status bar to display the
        provided message.

        :param message: str - The message to display.

        :return: None
        """
        status_bar.content.controls[1].controls[1].value = message
        update_control(status_bar)

    def update_scale(scale: float) -> None:
        """
        Update the scale of the viewer and image display.

        This function recalculates and updates the viewer and image scales based on the provided
        scale factor. It ensures that the viewer scale remains within a specified range, and it
        adjusts the image scale and viewer offset accordingly. The updated scale information is
        stored in the session and the status bar is refreshed to reflect the changes.

        :param scale: float - The factor by which to scale the current viewer scale.

        :return: None
        """
        # Execute only if the image is loaded
        if page.session.get('image_size'):
            # Get viewer and image sizes
            image_width, image_height = page.session.get('image_size')
            viewer_width, viewer_height = page.session.get('viewer_size')

            # Get starting viewer scale
            viewer_scale_0 = page.session.get('viewer_scale')

            # Calculate viewer scale (in range 1.0 - 15.0)
            viewer_scale = viewer_scale_0 * scale
            viewer_scale = max(1.0, min(15.0, viewer_scale))

            # Calculate window scale, image scale and scaled image size
            window_scale = min(viewer_width / image_width, viewer_height / image_height, 1.0)
            image_scale = window_scale * viewer_scale
            scaled_image_width = image_width * image_scale
            scaled_image_height = image_height * image_scale

            # Calculate main viewer offset (without moving and scale offset)
            main_viewer_offset = ft.Offset(
                x=round((viewer_width - scaled_image_width) / 2),
                y=round((viewer_height - scaled_image_height) / 2)
            )
            page.session.set('main_offset', main_viewer_offset)

            # Set viewer offset and margin for start position
            if not page.session.contains_key('viewer_offset'):
                page.session.set('viewer_offset', main_viewer_offset)
                page.session.set('margin', {
                    'left': 0,
                    'top': 0,
                    'right': 0,
                    'bottom': 0
                })

            # Update viewer, window and image scales
            page.session.set('viewer_scale', viewer_scale)
            page.session.set('window_scale', window_scale)
            page.session.set('image_scale', image_scale)
            page.session.set('scale', viewer_scale / viewer_scale_0)
            update_status_bar()

    def window_resized(e: ft.WindowResizeEvent) -> None:
        """
        Window resize event handler.

        This function is called whenever the window is resized. It stores the current
        viewer size in the session and updates the viewer size with the new window size.
        It then updates the viewer scale to fit the image in the window and refreshes
        the status bar.

        :param e: flet.WindowResizeEvent - The window event containing the new window size.

        :return: None
        """
        page.session.set('start_viewer_size', page.session.get('viewer_size'))

        # Set new viewer size with margin
        page.session.set('viewer_size', (e.width - 5 - 5, e.height - 40 - 5 - 5 - 26))

        update_scale(1)
        update_status_bar()

    def viewer_interaction_start(e: ft.InteractiveViewerInteractionStartEvent) -> None:
        """
        Viewer interaction start event handler.

        This function is called whenever the viewer interaction starts (like mouse click/move or
        zoom). It stores the current cursor position, viewer scale and viewer offset in the session.
        This data is used to calculate the new viewer offset and scale when the interaction ends.

        :param e: flet.InteractiveViewerInteractionStartEvent - The start event of the viewer.

        :return: None
        """
        # Current cursor coordinates relative to viewer
        local_x, local_y = e.local_focal_point.x, e.local_focal_point.y

        # Update the start cursor coordinates, start scale and start offset
        page.session.set('start_cursor', (local_x, local_y))
        page.session.set('start_scale', page.session.get('viewer_scale'))
        page.session.set('start_offset', page.session.get('viewer_offset'))

    def viewer_interaction_update(e: ft.InteractiveViewerInteractionUpdateEvent) -> None:
        """
        Viewer interaction update event handler.

        This function is triggered during an interaction update event within the viewer (like mouse
        move or zoom). It checks the scale of the event and updates the session with either the
        current focal point (for mouse move) or recalculates the viewer scale (for zoom).

        :param e: flet.InteractiveViewerInteractionUpdateEvent - The update event of the viewer.

        :return: None
        """
        if e.scale == 1:
            page.session.set('move', (e.local_focal_point.x, e.local_focal_point.y))
        else:
            update_scale(e.scale)

    def viewer_interaction_end(e: ft.InteractiveViewerInteractionEndEvent) -> None:
        """
        Viewer interaction end event handler.

        This function handles the end of an interaction with the viewer (such as mouse release
        or zoom end). It calculates and updates the viewer's offset and margins based on the
        interaction, for correct calculation of click coordinates relative to the real image scale.
        It also manages the moving and scale adjustments, updating the status message and bar
        accordingly.

        :param e: ft.InteractiveViewerInteractionEndEvent - The end event of the viewer interaction.

        :return: None
        """
        # Get values from session storage
        scale = page.session.get('scale')
        image_scale = page.session.get('image_scale')
        image_width, image_height = page.session.get('image_size')
        start_cursor_x, start_cursor_y = page.session.get('start_cursor')
        viewer_width, viewer_height = page.session.get('viewer_size')
        start_offset = page.session.get('start_offset')
        start_scale = page.session.get('start_scale')
        viewer_offset = page.session.get('viewer_offset')
        viewer_scale = page.session.get('viewer_scale')
        window_scale = page.session.get('window_scale')

        # Scale limit check
        scale_limit = start_scale == 15 and viewer_scale == 15

        # Calculate scaled image size
        scaled_image_width = image_width * image_scale
        scaled_image_height = image_height * image_scale

        # Calculate bounding box for image moving
        bounding_box_width = viewer_width * (viewer_scale - 1) + scaled_image_width
        bounding_box_height = viewer_height * (viewer_scale - 1) + scaled_image_height
        bx = (viewer_width - bounding_box_width) / 2
        by = (viewer_height - bounding_box_height) / 2

        bounds = {
            'left': round(bx, 0),
            'top': round(by, 0),
            'right': round(bx + bounding_box_width, 0),
            'bottom': round(by + bounding_box_height, 0)
        }

        # Moving
        if page.session.contains_key('move'):
            # Get move values
            move_x = page.session.get('move')[0] - start_cursor_x
            move_y = page.session.get('move')[1] - start_cursor_y

            # Get start margin
            margin_0 = page.session.get('margin')

            # Calculate new margin and viewer offset x,y coordinates
            margin_left = margin_0['left'] + move_x
            margin_top = margin_0['top'] + move_y
            margin_right = margin_0['right'] - move_x
            margin_bottom = margin_0['bottom'] - move_y

            vx = viewer_offset.x + move_x
            vy = viewer_offset.y + move_y

            if margin_left < 0:
                margin_left = 0
                margin_right = bounding_box_width - scaled_image_width
                vx = bounds['left']
            elif margin_right < 0:
                margin_right = 0
                margin_left = bounding_box_width - scaled_image_width
                vx = bounds['right'] - scaled_image_width
            if margin_top < 0:
                margin_top = 0
                margin_bottom = bounding_box_height - scaled_image_height
                vy = bounds['top']
            elif margin_bottom < 0:
                margin_bottom = 0
                margin_top = bounding_box_height - scaled_image_height
                vy = bounds['bottom'] - scaled_image_height

            # Set new margin and viewer offset x,y coordinates
            page.session.set('margin', {
                'left': margin_left,
                'top': margin_top,
                'right': margin_right,
                'bottom': margin_bottom
            })
            page.session.set('viewer_offset', ft.Offset(
                x=round(vx),
                y=round(vy)
            ))

            # Remove move values
            page.session.remove('move')

            # Update status message
            if not page.session.get('tool'):
                update_status_message("moving")

        # Scaling
        elif page.session.contains_key('scale') and not scale_limit:
            if round(image_scale, 5) == round(window_scale, 5): # Main scale state
                vx, vy = page.session.get('main_offset').x, page.session.get('main_offset').y
                margin_left, margin_top, margin_right, margin_bottom = 0, 0, 0, 0
            else:
                # Calculate the image offset when scaling
                x_scale_viewer_offset = (start_cursor_x - start_offset.x) * (1 - scale)
                y_scale_viewer_offset = (start_cursor_y - start_offset.y) * (1 - scale)

                _x = start_offset.x + x_scale_viewer_offset
                _y = start_offset.y + y_scale_viewer_offset

                # Calculate new margin and viewer offset x,y coordinates
                margin_left = _x - bounds['left']
                margin_top = _y - bounds['top']
                margin_right = bounds['right'] - (_x + scaled_image_width)
                margin_bottom = bounds['bottom'] - (_y + scaled_image_height)

                vx = viewer_offset.x + round(x_scale_viewer_offset)
                vy = viewer_offset.y + round(y_scale_viewer_offset)

                if margin_left < 0:
                    margin_left = 0
                    margin_right = bounding_box_width - scaled_image_width
                    vx = bounds['left']
                elif margin_right < 0:
                    margin_right = 0
                    margin_left = bounding_box_width - scaled_image_width
                    vx = bounds['right'] - scaled_image_width
                if margin_top < 0:
                    margin_top = 0
                    margin_bottom = bounding_box_height - scaled_image_height
                    vy = bounds['top']
                elif margin_bottom < 0:
                    margin_bottom = 0
                    margin_top = bounding_box_height - scaled_image_height
                    vy = bounds['bottom'] - scaled_image_height

            # Set new margin and viewer offset x,y coordinates
            page.session.set('margin', {
                'left': margin_left,
                'top': margin_top,
                'right': margin_right,
                'bottom': margin_bottom
            })
            page.session.set('viewer_offset', ft.Offset(
                x=round(vx),
                y=round(vy)
            ))

            # Remove scale values
            page.session.remove('scale')

            # Update status message
            if not page.session.get('tool'):
                update_status_message('scaling')

        # Click or scale limit handling
        else:
            # Scale limit handling
            if scale_limit:
                update_status_message('scale limit has been reached')
                update_status_bar()
                return

            # Get image scale and offset
            offset = page.session.get('viewer_offset')
            image_scale = page.session.get('image_scale')

            # Calucalte click coordinates relative to real image scale
            coord = [
                round((start_cursor_x - offset.x) / image_scale),
                round((start_cursor_y - offset.y) / image_scale)
            ]

            tool = page.session.get('tool')

            # Update status message for non-tool click
            if tool is None:
                update_status_message(f"click: (x={coord[0]}, y={coord[1]})")
                return

            # Click processing with the current tool
            if tool == 'Magic select':
                magic_select(coord)
            if tool == 'Manual select':
                manual_select(coord)
            if tool == 'Remove tool':
                remove_tool(coord)

        update_status_bar()

    def add_mezo_to_db(center: tuple, diameter: float) -> None:
        """
        Add a mezo to the database.

        The function takes a center point tuple (x, y) and a diameter float as input, and
        adds a record to the mezo_data table in the database. The image_id is retrieved
        from the current sample data in the session. The square value is calculated from
        the diameter.

        If the insertion is successful, the function retrieves the new mezo data from the
        database using get_mezo_data.

        :param center: tuple - Center point of the mezo (x, y)
        :param diameter: float - Diameter of the mezo

        :return: None
        """
        # Get image id
        image_id = page.session.get('sample_data')['images']\
            [page.session.get('image_index')]['image_id']

        # Calculate square
        square = 0.25 * math.pi * math.pow(diameter, 2)

        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            cur.execute(
                """INSERT INTO mezo_data
                (image_id, center_x, center_y, diameter, square)
                VALUES (?, ?, ?, ?, ?)""",
                (image_id, center[0], center[1], diameter, square)
            )
            conn.commit()

            # Update mezo data in session storage
            get_mezo_data(image_id)
        except sqlite3.Error as e:
            print(f"Error while adding mezo to database: {e}")
        finally:
            cur.close()
            conn.close()

    def add_mezo_to_viewer(x: float, y: float, radius: float, recursive: bool = False) -> None:
        """
        Add a mezo visualization to the viewer and manage mask history.

        This function draws a mezo (circular overlay) on the current image mask at the given
        coordinates and radius. The mask is updated and saved in session storage. If the `recursive`
        flag is False, the mask history is updated by saving the current mask state in a base64
        format, for faster "undo" operation. Only the last three masks are kept in history.

        :param x: float - The x-coordinate of the center of the mezo.
        :param y: float - The y-coordinate of the center of the mezo.
        :param radius: float - The radius of the mezo.
        :param recursive: bool - A flag indicating if the function is called recursively. If False,
                                the mask history is updated.

        :return: None
        """
        # Get current mask layer
        mask = page.session.get('mask')
        # Update mask history? if not recursive calling
        if not recursive:

            prev_masks = page.session.get('prev_masks')

            buffered = io.BytesIO()
            mask.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            prev_masks.append(mask_base64)

            if len(prev_masks) > 3:
                del prev_masks[0]

            page.session.set('prev_masks', prev_masks)

        # Get image size
        image_width, image_height = page.session.get('image_size')
        # Initialize overlay
        overlay = Image.new('RGBA', (image_width, image_height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay, mode='RGBA')

        # Calculate mezo circle coordinates
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        center_left_up_point = (x - 2, y - 2)
        center_right_down_point = (x + 2, y + 2)

        # Draw mezo
        fill_color = (191, 255, 0, 128)
        overlay_draw.ellipse(
            [left_up_point, right_down_point],
            fill=fill_color,
            outline='#BFFF00',
            width=3
        )
        overlay_draw.ellipse([center_left_up_point, center_right_down_point], fill='#BFFF00')

        # Add overlay to mask
        mask = Image.alpha_composite(mask, overlay)

        # Update mask in canvas
        buffered = io.BytesIO()
        mask.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        canvas.controls[0].controls[0].controls[0].content.content.controls[1] = \
            ft.Image(src_base64=img_base64)

        # Save new mask
        mask_dir = os.path.join(DATA_DIR, page.session.get('sample_data')['name'], 'masks')
        mask_path = os.path.join(mask_dir, f"mask {page.session.get('image_index') + 1}.png")
        mask.save(mask_path)

        # Update mask in session storage
        page.session.set('mask', mask)

        # Save result image
        save_image()

    def magic_select(coord: list) -> None:
        """
        Magic select tool.

        This function uses the SAM model to select a mezo from click coordinates, caluclate its
        center and radius, and add it to the database and mask layer.

        :param coord: list - X and Y coordinates of the user's click.

        :return: None
        """
        # Get SAM model
        predictor = page.session.get('sam_model')

        # Get predict
        sa_masks, sa_scores, sa_logits = predictor.predict(
            point_coords=np.array([coord]),
            point_labels=np.array([1]),
            multimask_output=False
        )

        # Calculate center and radius
        matrix = np.column_stack(np.where(sa_masks[0] > 0))
        center = matrix.mean(axis=0)
        distances = cdist([center], matrix)
        x, y = round(center[1]), round(center[0])
        radius = np.max(distances)

        # Add mezo to database
        add_mezo_to_db(
            center=(x, y),
            diameter=radius * 2
        )
        # Add mezo to mask layer
        add_mezo_to_viewer(
            x=x,
            y=y,
            radius=radius
        )

        # Update mask in canvas
        update_control(canvas.controls[0].controls[0].controls[0])

    def manual_select(coord: list) -> None:
        """
        Manual select tool.

        This function adds a mezo to the database and mask layer by manually selecting its
        center and diameter. The user must click on the center of the mezo first, then on the outer
        point. The diameter is calculated as the distance between the two clicks.

        :param coord: list - X and Y coordinates of the user's click.

        :return: None
        """
        x, y = coord[0], coord[1]
        image_width, image_height = page.session.get('image_size')

        # First click (center of the mezo)
        if not page.session.contains_key('current_coord'):
            # Save center coordinates in session storage
            page.session.set('current_coord', (x, y))
            # Initialize overlay
            overlay = Image.new('RGBA', (image_width, image_height), (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay, mode='RGBA')

            # Draw a center indicator with different size depending on the scale
            r = min(5, 5 // page.session.get('image_scale')) - 1
            if r > 0:
                overlay_draw.ellipse([x - r, y - r, x + r, y + r], fill='red')
            else:
                overlay_draw.point((x, y), fill='red')

            # Add overlay to mask
            buffered = io.BytesIO()
            overlay.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            canvas.controls[0].controls[0].controls[0].content.content.controls.append(ft.Image(src_base64=img_base64))

            # Update status message (info for next step - second click)
            update_status_message("Click on the outer point of the mezophase to set the diameter.")

        # Second click (outer point of the mezo)
        else:
            # Remove center indicator
            del canvas.controls[0].controls[0].controls[0].content.content.controls[-1]

            # Calculate center and radius
            x0, y0 = page.session.get('current_coord')
            radius = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)

            # Remove center coordinates
            page.session.remove('current_coord')

            # Add mezo to database
            add_mezo_to_db(
                center=(x0, y0),
                diameter=radius * 2
            )
            # Add mezo to mask layer
            add_mezo_to_viewer(
                x=x0,
                y=y0,
                radius=radius
            )

            # Update status message (info for next step - first click)
            update_status_message("Click on the center of mezophase.")

        # Update mask in canvas
        update_control(canvas.controls[0].controls[0].controls[0])

    def remove_tool(coord: list) -> None:
        """
        Remove tool.

        This function is called when the user clicks on the remove tool icon in the toolbar.
        It finds the mezo that is closest to the click coordinates and removes it from the mask
        layer and the database.

        :param coord: list - X and Y coordinates of the user's click.

        :return: None
        """
        x, y = coord[0], coord[1]
        mezo_to_remove = None

        # Search for the mezo selected for removing
        for mezo in list(page.session.get('mezo_data').values())[::-1]:
            x0, y0 = mezo['center']
            radius = mezo['diameter'] / 2

            # Euclidean distance check
            if math.sqrt((x - x0)**2 + (y - y0)**2) <= radius:
                mezo_to_remove = mezo['mezo_id']
                break

        # Remove mezo
        remove_mezo(key=mezo_to_remove)

    def open_image(sample_id: int, image_index: int) -> None:
        """
        Open an image and initialize viewer settings.

        This function initializes a new sample image by updating the session store, creating a
        viewer and updating other widgets.

        :param sample_id: int - The ID of the sample to open.
        :param image_index: int - The index of the image in the sample to open.

        :return: None
        """
        # Clear session storage
        if page.session.contains_key('viewer_offset'):
            page.session.remove('viewer_offset')
        if page.session.contains_key('tool'):
            page.session.remove('tool')
            toolbar.selected_index = None
            update_control(toolbar)
        if page.session.contains_key('start_cursor'):
            page.session.remove('start_cursor')
        if page.session.contains_key('start_scale'):
            page.session.remove('start_scale')
        if page.session.contains_key('start_offset'):
            page.session.remove('start_offset')
        if page.session.contains_key('current_coord'):
            page.session.remove('current_coord')
        if page.session.contains_key('sam_model'):
            page.session.remove('sam_model')

        # Get sample data
        get_sample_data(sample_id)

        # Get image data
        image_id = page.session.get('sample_data')['images'][image_index]['image_id']
        get_mezo_data(image_id)

        # Paths to image and mask
        sample_name = page.session.get('sample_data')['name']
        image_dir = os.path.join(DATA_DIR, sample_name, 'low')
        image_path = os.path.join(image_dir, os.listdir(image_dir)[image_index])
        mask_dir = os.path.join(DATA_DIR, sample_name, 'masks')
        mask_path = os.path.join(mask_dir, f"mask {image_index + 1}.png")

        # Set image and mask paths, image size and mask (Image class instance)
        page.session.set('image_path', image_path)
        page.session.set('mask_path', mask_path)
        image_pil = Image.open(page.session.get('image_path'))
        page.session.set('image_size', (image_pil.width, image_pil.height))
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = Image.new('RGBA', (image_pil.width, image_pil.height), (255, 255, 255, 0))
        page.session.set('mask', mask)
        image_pil.close()

        # Initialize viewer scale and offset
        page.session.set('viewer_scale', 1.0)
        window_resized(page)

        # Initialize viewer
        viewer = ft.InteractiveViewer(
            expand=True,
            min_scale=None,
            max_scale=15,
            boundary_margin=ft.margin.all(0),
            scale_factor=1500.0,
            content=ft.Container(
                ft.Stack(
                    [
                        ft.Image(page.session.get('image_path')),
                        ft.Image(mask_path)
                    ]
                ),
                alignment=ft.alignment.center
            ),
            interaction_end_friction_coefficient=0.0,
            on_interaction_start=viewer_interaction_start,
            on_interaction_update=viewer_interaction_update,
            on_interaction_end=viewer_interaction_end,
        )

        # Adding viewer to canvas
        canvas.controls[0].controls[0].controls[0] = viewer

        # Adding toolbar to canvas
        canvas.controls[0].controls.append(
            ft.Container(
                toolbar,
                height=150,
                border_radius=20,
                margin=ft.margin.only(right=10),
                expand=True
            )
        )

        # Adding mezo list button to canvas
        mezo_list_button = ft.Container(
            ft.IconButton(
                icon=ft.Icons.LIST_ALT,
                tooltip='Открыть таблицу',
                icon_color='white',
                highlight_color=COLOR_SCHEME['highlight'],
                on_click=open_results_table
            ),
            bgcolor=COLOR_SCHEME['secondary'],
            border_radius=20,
            height=50,
            margin=ft.margin.only(left=10),
            padding=ft.padding.all(5)
        )
        canvas.controls.append(mezo_list_button)

        # Get scale and porosity values and update relevant dialog widgets
        scale_px = page.session.get('sample_data')['images'][image_index]['scale_px']
        scale_mkm = page.session.get('sample_data')['images'][image_index]['scale_mkm']
        scale_changer.content.controls[0].controls[0].controls[0].value = scale_px
        scale_changer.content.controls[0].controls[2].controls[0].value = scale_mkm
        porosity_changer.content.controls[0].value = \
            page.session.get('sample_data')['images'][image_index]['porosity']

        # Update status bar
        status_bar.content.controls[2].controls[
            1].value = f"{page.session.get('image_size')[0]}x{page.session.get('image_size')[1]} px"
        update_status_message('')
        update_status_bar()

        # Update menubar
        menubar.content.controls[1].controls[0].value = page.session.get('sample_data')['name']
        menubar.content.controls[2].controls[1].value = os.listdir(image_dir)[image_index]

        # Initialize mask history (for 'undo')
        page.session.set('prev_masks', [])

        # Save result image
        save_image()

    def save_image() -> None:
        """
        Save the current image and mask as a result image.

        This function takes image and mask paths from the session storage, opens
        the image and mask, and saves the result image (image with mask) to the
        result directory. If the mask path does not exist, the function does nothing.

        :return: None
        """
        # Get paths
        sample_name = page.session.get('sample_data')['name']
        result_dir = os.path.join(DATA_DIR, sample_name, 'result')
        image_path = page.session.get('image_path')
        mask_path = page.session.get('mask_path')

        # If mask is not existing, return None
        if not os.path.exists(mask_path):
            return

        # Opening images and merging them
        image = Image.open(image_path).convert('RGBA')
        mask = Image.open(mask_path).convert('RGBA')
        image.paste(mask, (0, 0), mask)

        # Save result image
        image.convert('RGB').save(os.path.join(result_dir, os.path.basename(image_path)))

        image.close()
        mask.close()

    def arrow_styling(image_index: int, image_count: int) -> None:
        """
        Change the style of the left and right arrow buttons in the menubar.

        This function takes the current image index and the total count of images and updates
        the style of the left and right arrow buttons in the menubar when the first/last image
        is reached.

        :param image_index: The current image index
        :param image_count: The total count of images

        :return: None
        """
        left_arrow_disabled, right_arrow_disabled = False, False
        left_arrow_color, right_arrow_color = COLOR_SCHEME['primary'], COLOR_SCHEME['primary']

        # First image
        if image_index == 0:
            left_arrow_disabled = True
            left_arrow_color = COLOR_SCHEME['primary'] + ',0.25'
        # Last image
        elif image_index == image_count - 1:
            right_arrow_disabled = True
            right_arrow_color = COLOR_SCHEME['primary'] + ',0.25'

        # Update arrow widgets
        menubar.content.controls[2].controls[0].disabled = left_arrow_disabled
        menubar.content.controls[2].controls[0].icon_color = left_arrow_color
        menubar.content.controls[2].controls[2].disabled = right_arrow_disabled
        menubar.content.controls[2].controls[2].icon_color = right_arrow_color

    def previous_image(e: ft.ControlEvent) -> None:
        """
        Handle the event of clicking the previous image button.

        This function decreases the image index by 1 (but not less than 0), updates the arrow
        styling, opens the new image and updates the page.

        :param e: flet.ControlEvent - The event containing control details.

        :return: None
        """
        image_index = page.session.get('image_index')
        image_count = page.session.get('sample_data')['count']
        if image_index > 0:
            image_index -= 1
            page.session.set('image_index', image_index)
            arrow_styling(image_index, image_count)
            open_image(page.session.get('sample_id'), image_index)
            page.update()

    def next_image(e: ft.ControlEvent) -> None:
        """
        Handle the event of clicking the next image button.

        This function increments the image index by 1 (but not more than the number of images in
        the sample - 1), updates the arrow styling, opens the new image and updates the page.

        :param e: flet.ControlEvent - The event containing control details.

        :return: None
        """
        image_index = page.session.get('image_index')
        image_count = page.session.get('sample_data')['count']
        if image_index < image_count - 1:
            image_index += 1
            page.session.set('image_index', image_index)
            arrow_styling(image_index, image_count)
            open_image(page.session.get('sample_id'), image_index)
            page.update()

    def update_image_data(
            image_index: 'int',
            parameter: 'str',
            value: Union['int', 'float', 'tuple']
        ) -> None:
        """
        Update the database with a parameter for a specific image.

        This function takes an image index, a parameter and a value as input, and updates the
        corresponding entry in the images table in the database. If the parameter does not
        exist, the function returns None.

        :param image_index: int - The index of the image to update.
        :param parameter: str - The parameter to update. Can be 'porosity' or 'scale_px_mkm'.
        :param value: int, float, tuple - The value to update the parameter with. If parameter is
                                          'porosity', value should be a float between 0 and 100.
                                          If parameter is 'scale_px_mkm', value should be a tuple
                                          of two floats.

        :return: None
        """
        image_id = page.session.get('sample_data')['images'][image_index]['image_id']

        # Queries for different parameters
        if parameter == 'porosity':
            upd_params = (value / 100, image_id)
            query = "UPDATE images SET porosity = ? WHERE image_id = ?"
            print()

        elif parameter == 'scale_px_mkm':
            scale_px, scale_mkm = value
            upd_params = (scale_px, scale_mkm, image_id)
            query = "UPDATE images SET scale_px = ?, scale_mkm = ? WHERE image_id = ?"
        else:
            return

        # Update database
        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            cur.execute(
                query,
                upd_params
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error while updating porosity: {e}")
        finally:
            cur.close()
            conn.close()

        # Update sample data in session storage
        get_sample_data(page.session.get('sample_data')['sample_id'])

    def open_results_table(e: ft.ControlEvent) -> None:
        """
        Open the results table with data on the current image.

        This function opens a navigation drawer with a table containing data on the
        current image. The table includes the number of mezophases, maximum size of mezophases,
        porosity of the material, area of the material, area of mezophases and
        the percentage of mezophases.

        :param e: flet.ControlEvent - The event containing control details.

        :return: None
        """
        # Initialize drawer
        mezo_list = ft.NavigationDrawer(
            controls=[
                ft.Container(
                    ft.Text(
                        'Результаты анализа',
                        size=16,
                        style=ft.TextStyle(weight=ft.FontWeight.BOLD)
                    ),
                    alignment=ft.alignment.center,
                    padding=ft.padding.only(top=20, bottom=20)
                ),
                ft.Divider(),
                ft.Container(ft.Column(
                    [
                        ft.Row([ft.Markdown(f"Количество мезофаз: **{0}**")]),
                        ft.Row([ft.Markdown(f"Максимальный размер: **{0} мкм** ")]),
                        ft.Row([ft.Markdown(f"Пористость материала: **{0} %**")]),
                        ft.Row([ft.Markdown(f"Площадь материала: **{0} мм^2**")]),
                        ft.Row([ft.Markdown(f"Площадь мезофазы: **{0} мм^2**")]),
                        ft.Row([ft.Markdown(f"Процентное содержание: **{0} %**")])
                    ], spacing=20),
                    padding=ft.padding.all(10)
                ),
                ft.Divider(),
                ft.DataTable(
                    columns=[
                        ft.DataColumn(ft.Text("№")),
                        ft.DataColumn(ft.Text("D, мкм")),
                        ft.DataColumn(ft.Text("S, мкм2"), numeric=True),
                    ],
                    rows=[],
                )
            ],
            bgcolor=COLOR_SCHEME['background']
        )

        # Add drawer on current page view
        page.views[-1].drawer = mezo_list

        # Get mezo, scale and porosity data
        image_data = page.session.get('sample_data')['images'][page.session.get('image_index')]
        mezo_data = page.session.get('mezo_data')
        scale_factor = image_data['scale_mkm'] / image_data['scale_px']
        porosity = image_data['porosity']

        # Add rows in result table
        mezo_list.controls[4].rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(int(key) + 1)),
                    ft.DataCell(ft.Text(f"{d['diameter'] * scale_factor:.2f}")),
                    ft.DataCell(ft.Text(f"{d['square'] * scale_factor ** 2:.2f}")),
                ],
            ) for key, d in mezo_data.items()
        ]

        image_width, image_height = page.session.get('image_size')

        # Calculate analytical parameters
        max_size = max([d['diameter'] for d in mezo_data.values()]) * scale_factor \
            if len(mezo_data) > 0 else 0
        material_square = (1 - porosity) * image_width * image_height * 0.000001 * scale_factor ** 2
        total_square = sum([d['square'] for d in mezo_data.values()]) * 0.000001 * scale_factor ** 2
        mezo_percentage = total_square / material_square

        # Add analytical parameters to drawer
        mezo_list.controls[2].content.controls[0].controls[0].value = \
            f"Количество мезофаз: **{len(mezo_data)}**"
        mezo_list.controls[2].content.controls[1].controls[0].value = \
            f"Максимальный размер: **{max_size:.2f} мкм** "
        mezo_list.controls[2].content.controls[2].controls[0].value = \
            f"Пористость материала: **{porosity:.2%}**"
        mezo_list.controls[2].content.controls[3].controls[0].value = \
            f"Площадь материала: **{material_square:.3f} мм^2**"
        mezo_list.controls[2].content.controls[4].controls[0].value = \
            f"Площадь мезофазы: **{total_square:.3f} мм^2**"
        mezo_list.controls[2].content.controls[5].controls[0].value = \
            f"Процентное содержание: **{mezo_percentage:.2%}**"

        # Open drawer
        page.open(mezo_list)

    def change_porosity(e: ft.ControlEvent) -> None:
        """
        Change porosity of material.

        This function is called when the 'Ok' button is clicked in the 'Change porosity' dialog.
        It updates the porosity of the material in the database and closes the dialog.

        :param e: flet.ControlEvent - The event containing control details.
        :return: None
        """
        # If porosity field is empty - return
        if page.session.get('porosity_field') is None:
            page.session.remove('porosity_field')
            page.close(porosity_changer)
            return

        update_image_data(
            image_index=page.session.get('image_index'),
            parameter='porosity',
            value=float(page.session.get('porosity_field'))
        )
        page.session.remove('porosity_field')
        page.close(porosity_changer)

    def change_scale_factor(e: ft.ControlEvent) -> None:
        """
        Change the scale factor of the image.

        This function is triggered when the 'Ok' button is clicked in the 'Change scale' dialog.
        It updates the scale factor of the image in the database using the values from the session
        fields for pixels and micrometers, and then removes these fields from the session.
        The status bar is updated to reflect the new scale, and the dialog is closed.

        :param e: flet.ControlEvent - The event containing control details.
        :return: None
        """
        update_image_data(
            image_index=page.session.get('image_index'),
            parameter='scale_px_mkm',
            value=(
                int(page.session.get('px_field')),
                int(page.session.get('mkm_field'))
            )
        )
        page.session.remove('px_field')
        page.session.remove('mkm_field')
        update_status_bar()
        page.close(scale_changer)

    def sam_init() -> None:
        """
        Initialize the SAM model.

        This function is called when the SAM model is needed, but it is not yet initialized.
        It creates a SAM model using the model type and checkpoint from the session and creates
        a predictor from the model. It then reads the image from the session, converts it to RGB
        and fit predictor on them. Finally, it saves the predictor to the session and
        updates the SAM model state.

        :return: None
        """
        if not page.session.contains_key('sam_model'):
            # Load model checkpoint
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            # Initialize predictor
            predictor = SamPredictor(sam)

            # Load image
            sa_image = cv2.imdecode(
                np.fromfile(page.session.get('image_path'), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            sa_image = cv2.cvtColor(sa_image, cv2.COLOR_BGR2RGB)

            # Fit predictor
            predictor.set_image(sa_image)

            # Save predictor in session storage
            page.session.set('sam_model', predictor)

    def change_tool(e: Union[ft.ControlEvent, int]) -> None:
        """
        Change the current tool in the session based on the user's selection.

        This function takes a control event, determines the selected tool index,
        and updates the session with the corresponding tool name. It also handles
        status messages and model initialization for the 'Magic select' tool.

        :param e: flet.ControlEvent, int - The event containing control details, or an integer index
                                           indicating the selected tool (from keyboard).

        :return: None
        """
        # Keyboard input
        if isinstance(e, int):
            index = e
        # Widget input
        else:
            index = e.control.selected_index

        # Set selected tool
        tools = ['Magic select', 'Manual select', 'Remove tool']
        page.session.set('tool', tools[index])

        # Update status message for selected tool
        if page.session.get('tool') == 'Magic select':
            if page.session.contains_key('current_coord'):
                page.session.remove('current_coord')
                del canvas.controls[0].controls[0].controls[0].content.content.controls[-1]
                update_control(canvas.controls[0].controls[0].controls[0])

            update_status_message(
                "The model is being prepared. Please wait and don't press any buttons."
            )
            sam_init()
            update_status_message('Model is ready!')
        elif page.session.get('tool') == 'Manual select':
            if page.session.contains_key('current_coord'):
                page.session.remove('current_coord')
                del canvas.controls[0].controls[0].controls[0].content.content.controls[-1]
                update_control(canvas.controls[0].controls[0].controls[0])

            update_status_message("Click on the center of mezophase.")
        elif page.session.get('tool') == 'Remove tool':
            if page.session.contains_key('current_coord'):
                page.session.remove('current_coord')
                del canvas.controls[0].controls[0].controls[0].content.content.controls[-1]
                update_control(canvas.controls[0].controls[0].controls[0])

            update_status_message("Click on mezophase to remove it.")

    def reset_viewer(e: ft.ControlEvent) -> None:
        """
        Reset the viewer to the default state.

        This function is called whenever the user clicks the 'Reset viewer' button. It resets the
        viewer control to its default state and updates the session with the default viewer
        scale, image scale, and removes any stored viewer offset and start values. It then
        updates the status bar and calls the window_resized function to update the offset
        and margin.

        :param e: flet.ControlEvent - The event containing control details.

        :return: None
        """
        # Reset viewer
        canvas.controls[0].controls[0].controls[0].reset()

        # Update session storage
        page.session.set('viewer_scale', 1)
        page.session.set('image_scale', page.session.get('window_scale'))
        if page.session.contains_key('viewer_offset'):
            page.session.remove('viewer_offset')
        if page.session.contains_key('start_cursor'):
            page.session.remove('start_cursor')
        if page.session.contains_key('start_scale'):
            page.session.remove('start_scale')
        if page.session.contains_key('start_offset'):
            page.session.remove('start_offset')

        # Update offset and margin
        window_resized(page)

        # Update status bar
        update_status_bar()

    def on_keyboard(e: ft.KeyboardEvent) -> None:
        """
        Keyboard event handler.

        This function is called whenever a keyboard event occurs in the page. It handles
        the following events:

        - Escape: If the 'current_coord' is present in the session, it removes the current
                   coordinate and all its associated widgets (cancel selected center of mezo). If
                   not, it sets the selected index of the toolbar to None and resets the tool to
                   None (cancel selected tool).
        - 1: switch to 'Magic select' tool'.
        - 2: switch to 'Manual select' tool'.
        - 3: switch to 'Remove tool' tool'.
        - Ctrl+Z: If the 'mezo_data' is present in the session and the 'current_coord' is
                   not present, it calls the remove_mezo function ('undo' operation).

        :param e: ft.KeyboardEvent - The event containing keyboard details.

        :return: None
        """
        # Cancel selected center of mezo or selected tool
        if e.key == 'Escape':
            if page.session.contains_key('current_coord'):
                page.session.remove('current_coord')
                del canvas.controls[0].controls[0].controls[0].content.content.controls[-1]
                update_control(canvas.controls[0].controls[0].controls[0])
            else:
                toolbar.selected_index = None
                page.session.set('tool', None)
                update_control(toolbar)

        #  Switch to Magic select
        if e.key == '1':
            toolbar.selected_index = 0
            change_tool(0)
            update_control(toolbar)

        # Switch to Manual select
        if e.key == '2':
            toolbar.selected_index = 1
            change_tool(1)
            update_control(toolbar)

        # Switch to Remove tool
        if e.key == '3':
            toolbar.selected_index = 2
            change_tool(2)
            update_control(toolbar)

        # Undo
        if e.key == 'Z' and e.ctrl:
            ready_to_drop = page.session.contains_key('mezo_data') and \
                            not page.session.contains_key('current_coord') and \
                            len(page.session.get('mezo_data')) > 0
            if ready_to_drop:
                remove_mezo()

    def remove_mezo(key: Union[str, int, None] = 'undo') -> None:
        """
        Remove mezo from the mask and mezo_data table in the db.

        :param key: str, int, None - The key to remove from the mezo_data table. If 'undo',
                                the last mezo is removed, otherwise the mezo with the given id
                                is removed.
        :return: None
        """
        def distance(x1: float, y1: float, x2: float, y2: float) -> float:
            """
            Calculate the Euclidean distance between two points.

            :param x1: float - X coordinate of the first point.
            :param y1: float - Y coordinate of the first point.
            :param x2: float - X coordinate of the second point.
            :param y2: float - Y coordinate of the second point.

            :return: float - The Euclidean distance between the two points.
            """
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        def circles_intersect(mezo1: tuple, mezo2: tuple) -> bool:
            """
            Check if two circles intersect.

            :param mezo1: tuple - Tuple of (x, y, radius) for the first circle.
            :param mezo2: tuple - Tuple of (x, y, radius) for the second circle.

            :return: bool - True if the circles intersect, False otherwise.
            """
            x1, y1, r1 = mezo1
            x2, y2, r2 = mezo2
            dist = distance(x1, y1, x2, y2)
            return dist < (r1 + r2)

        def remove_mezo_from_viewer(mezo_to_removed: tuple) -> None:
            """
            Remove a mezo from the viewer.

            This function takes a tuple mezo_to_removed containing the mezo's id, x/y coordinates,
            and radius. It removes the mezo from the mask layer and updates the viewer by drawing
            the new mask on the canvas. The mask image is also saved in the 'masks' folder of
            the sample's directory.

            :param mezo_to_removed: tuple - Tuple containing the mezo's id, x and y coordinates,
                                            and radius.

            :return: None
            """
            x, y, radius = mezo_to_removed[2], mezo_to_removed[3], mezo_to_removed[4] / 2
            mask = page.session.get('mask')
            mask_draw = ImageDraw.Draw(mask, mode='RGBA')

            # Make mezo pixels full transparent
            left_up_point = (x - radius, y - radius)
            right_down_point = (x + radius, y + radius)
            mask_draw.ellipse([left_up_point, right_down_point], fill=(255, 255, 255, 0))

            # Update viewer
            buffered = io.BytesIO()
            mask.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            canvas.controls[0].controls[0].controls[0].content.content.controls[1] = \
                ft.Image(src_base64=img_base64)

            # Save new mask
            mask_dir = os.path.join(DATA_DIR, page.session.get('sample_data')['name'], 'masks')
            mask_path = os.path.join(mask_dir, f"mask {page.session.get('image_index') + 1}.png")
            mask.save(mask_path)

            # Update mask in session storage
            page.session.set('mask', mask)

        def recursive_removing(start_mezo: tuple, all_mezo: list) -> None:
            """
            Recursively remove overlapping mezo from the viewer.

            This function takes a tuple start_mezo containing the mezo's id, x/y coordinates, and
            radius, and a list all_mezo containing all existing mezo of the current image. It
            checks if any of the existing mezo overlap with the start_mezo, and if so, removes the
            overlapping mezo from the viewer by calling remove_mezo_from_viewer. The function then
            calls itself with the newly removed mezo and the updated list of all mezo.

            :param start_mezo: tuple - Tuple containing the mezo's id, x and y coordinates,
                                       and radius.
            :param all_mezo: list - List of tuples containing the id, x and y coordinates,
                                    and radius for all existing mezo of the current image.

            :return: None
            """
            for i, row in enumerate(all_mezo):
                if circles_intersect(
                    mezo1=(start_mezo[2], start_mezo[3], start_mezo[4] / 2),
                    mezo2=(row[2], row[3], row[4] / 2)
                ):
                    new_all_mezo = all_mezo.copy()
                    new_iter_row = new_all_mezo.pop(i)
                    remove_mezo_from_viewer(row)
                    recursive_removing(new_iter_row, new_all_mezo)

        def recursive_adding(start_mezo: tuple, all_mezo: list) -> None:
            """
            Recursively add overlapping mezo to the viewer.

            This function takes a tuple start_mezo containing the mezo's id, x/y coordinates,
            and radius, and a list all_mezo containing all existing mezo of the current image.
            It checks if any of the existing mezo overlap with the start_mezo, and if so, adds the
            overlapping mezo to the viewer by calling add_mezo_to_viewer. The function then
            calls itself with the newly added mezo and the updated list of all mezo.

            This is necessary to correctly display the overlapping circles after removing the mezo.

            :param start_mezo: tuple - Tuple containing the mezo's id, x and y coordinates,
                                    and radius.
            :param all_mezo: list - List of tuples containing the id, x and y coordinates,
                                    and radius for all existing mezo of the current image.

            :return: None
            """
            for i, row in enumerate(all_mezo):
                if circles_intersect(
                    mezo1=(start_mezo[2], start_mezo[3], start_mezo[4] / 2),
                    mezo2=(row[2], row[3], row[4] / 2)
                ):
                    new_all_mezo = all_mezo.copy()
                    new_iter_row = new_all_mezo.pop(i)
                    add_mezo_to_viewer(row[2], row[3], row[4] / 2, recursive=True)
                    recursive_adding(new_iter_row, new_all_mezo)

        if not key:
            return

        image_id = page.session.get('sample_data')['images']\
            [page.session.get('image_index')]['image_id']

        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            # Undo operation
            if key == 'undo':
                cur.execute(
                    "SELECT * FROM mezo_data WHERE image_id = ? ORDER BY mezo_id DESC LIMIT 1",
                    (image_id,)
                )

            # Removal of a specific mezo with the `Remove tool`
            else:
                cur.execute("SELECT * FROM mezo_data WHERE mezo_id = ?", (key,))

            mezo_to_remove = cur.fetchone()

            # Delete the selected mezo from database
            cur.execute("DELETE FROM mezo_data WHERE mezo_id = ?", (mezo_to_remove[0],))
            conn.commit()

            # Update mezo dat in session storage
            get_mezo_data(image_id)

            # Get all mezo data from database
            cur.execute("SELECT * FROM mezo_data WHERE image_id = ?", (image_id,))
            rows = cur.fetchall()

            # Get previous mask
            prev_masks = page.session.get('prev_masks')

            # For the undo operation, use the previous masks, if any exist
            if len(prev_masks) > 0 and key == 'undo':

                img_base64 = prev_masks.pop(-1)
                canvas.controls[0].controls[0].controls[0].content.content.controls[1] = \
                    ft.Image(src_base64=img_base64)

                page.session.set('prev_masks', prev_masks)

                image_data = base64.b64decode(img_base64)
                image_bytes = io.BytesIO(image_data)
                last_mask = Image.open(image_bytes)

                page.session.set('mask', last_mask)
                last_mask.save(page.session.get('mask_path'))
            # Otherwise, remove selected mezo recursively
            else:
                remove_mezo_from_viewer(mezo_to_remove)
                recursive_removing(mezo_to_remove, rows)
                recursive_adding(mezo_to_remove, rows)

            # Update viewer and save new result image
            update_control(canvas.controls[0].controls[0].controls[0])
            save_image()
        except sqlite3.Error as e:
            print(f"Error while removing mezo: {e}")
        finally:
            cur.close()
            conn.close()

    def delete_sample() -> None:
        """
        Delete a sample from the database and file system.

        This function deletes all images, mezo data, and sample data associated with the
        current sample, and removes the sample's directory from the file system. It then
        navigates to the library page.

        :return: None
        """
        conn = sqlite3.connect(MEZO_DB)
        cur = conn.cursor()
        try:
            # Get sample data
            sample_id = page.session.get('sample_data')['sample_id']
            sample_path = os.path.join(DATA_DIR, page.session.get('sample_data')['name'])
            image_ids = [d['image_id'] for d in page.session.get('sample_data')['images'].values()]

            # Go to library view
            page.go('/library')

            # Delete sample from database
            cur.execute(
                f"DELETE FROM mezo_data WHERE image_id in ({', '.join('?' for _ in image_ids)})",
                image_ids
            )
            cur.execute("DELETE FROM images WHERE sample_id = ?", (sample_id,))
            cur.execute("DELETE FROM samples WHERE sample_id = ?", (sample_id,))
            conn.commit()

            # Close mask Image-file
            page.session.get('mask').close()

            # Recurcive removing sample directory
            shutil.rmtree(sample_path)
        except sqlite3.Error as e:
            print(f"Error while deleting sample: {e}")
        finally:
            cur.close()
            conn.close()

    def to_library(e: ft.ControlEvent) -> None:
        """
        Navigate to the library page.

        :param e: flet.ControlEvent - The event that triggered this function.

        :return: None
        """
        page.go('/library')

    page.on_resized = window_resized
    page.on_keyboard_event = on_keyboard

    # # UI APPLICATION CONTROLS
    # About dialog
    about = ft.AlertDialog(
        title=ft.Text('Mezo', text_align=ft.TextAlign.CENTER),
        bgcolor=COLOR_SCHEME['background'],
        content=ft.Column(
            [
                ft.Text('v.0.01\n\n\n'),
                ft.Text(
                    'Приложение находится в стадии тестирования.\n\n\n',
                    text_align=ft.TextAlign.CENTER
                ),
                ft.Text(
                    'Автор: Роман Козлов',
                    text_align=ft.TextAlign.CENTER
                ),
                ft.Markdown(
                    '[onemoreuselessthing.com](https://project11648075.tilda.ws/)',
                    on_tap_link=lambda e: page.launch_url(e.data)
                ),
                ft.Markdown(
                    '[Github](https://github.com/donatorex)',
                    on_tap_link=lambda e: page.launch_url(e.data)
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=250
        )
    )
    # Scale changer dialog
    scale_changer = ft.AlertDialog(
        title=ft.Text('Задать масштаб', text_align=ft.TextAlign.CENTER),
        bgcolor=COLOR_SCHEME['background'],
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.TextField(
                                    label='Пиксели',
                                    suffix_text='px',
                                    border_color=COLOR_SCHEME['secondary'],
                                    on_change=lambda e: page.session.set('px_field', e.data)
                                )
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            width=100
                        ),
                        ft.Column(
                            [ft.Text('=')],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            width=50
                        ),
                        ft.Column(
                            [
                                ft.TextField(
                                    label='Микроны',
                                    suffix_text='мкм',
                                    border_color=COLOR_SCHEME['secondary'],
                                    on_change=lambda e: page.session.set('mkm_field', e.data)
                                )
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            width=100
                        )
                    ],
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                ft.ElevatedButton(
                    content=ft.Text('OK', text_align=ft.TextAlign.CENTER),
                    style=ft.ButtonStyle(
                        color=COLOR_SCHEME['background'],
                        bgcolor=COLOR_SCHEME['highlight'],
                        overlay_color='#B098F9',
                    ),
                    on_click=change_scale_factor
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=30,
            height=100,
            width=300
        )
    )
    # Porosity changer dialog
    porosity_changer = ft.AlertDialog(
        title=ft.Text('Задать пористость', text_align=ft.TextAlign.CENTER),
        bgcolor=COLOR_SCHEME['background'],
        content=ft.Column(
            [
                ft.TextField(
                    label='Пористость',
                    suffix_text='%',
                    border_color=COLOR_SCHEME['secondary'],
                    on_change=lambda e: page.session.set('porosity_field', e.data)
                ),
                ft.ElevatedButton(
                    content=ft.Text('OK', text_align=ft.TextAlign.CENTER),
                    style=ft.ButtonStyle(
                        color=COLOR_SCHEME['background'],
                        bgcolor=COLOR_SCHEME['highlight'],
                        overlay_color='#B098F9',
                    ),
                    on_click=change_porosity
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=30,
            height=100,
            width=300
        )
    )
    # Delete sample dialog
    delete_sample_dialog = ft.AlertDialog(
        title=ft.Text('Подтвердить удаление', text_align=ft.TextAlign.CENTER),
        bgcolor=COLOR_SCHEME['background'],
        content=ft.Column(
            [
                ft.Container(
                    ft.Text(
                        'Вы действительно хотите удалить образец вместе со всеми снимками и результатами анализа?',  # noqa: E501
                        text_align=ft.TextAlign.CENTER
                    ),
                    alignment=ft.alignment.center,
                    height=70
                ),
                ft.ElevatedButton(
                    content=ft.Text('Подтвердить', text_align=ft.TextAlign.CENTER),
                    style=ft.ButtonStyle(
                        color=COLOR_SCHEME['background'],
                        bgcolor=COLOR_SCHEME['highlight'],
                        overlay_color='#B098F9',
                    ),
                    on_click=lambda _: delete_sample()
                )

            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=30,
            height=130,
            width=300
        ),
    )
    # Menubar
    menubar = ft.Container(
        ft.Row(
            [
                ft.MenuBar(
                    [
                        ft.Row(
                            [
                                ft.SubmenuButton(
                                    content=ft.Text('Файл'),
                                    controls=[
                                        ft.MenuItemButton(
                                            content=ft.Text('Импортировать JSON')
                                        ),
                                        ft.MenuItemButton(
                                            content=ft.Text('Экспортировать JSON')
                                        ),
                                        ft.Divider(),
                                        ft.MenuItemButton(
                                            content=ft.Text('Сформировать отчет'),
                                            on_click=lambda _: create_pdf_report(
                                                page.session.get('sample_id')
                                            )
                                        ),
                                        ft.Divider(),
                                        ft.MenuItemButton(
                                            content=ft.Text('Настройки')
                                        ),
                                        ft.Divider(),
                                        ft.MenuItemButton(
                                            content=ft.Text('Удалить образец', color='#B90000'),
                                            on_click=lambda e: page.open(delete_sample_dialog)
                                        ),
                                    ],
                                ),
                                ft.SubmenuButton(
                                    content=ft.Text('Редактировать'),
                                    controls=[
                                        ft.MenuItemButton(
                                            content=ft.Text('Отменить'),
                                        ),
                                        ft.MenuItemButton(
                                            content=ft.Text('Повторить')
                                        ),
                                        ft.Divider(),
                                        ft.MenuItemButton(
                                            content=ft.Text('Задать масштаб'),
                                            on_click=lambda e: page.open(scale_changer)
                                        ),
                                        ft.MenuItemButton(
                                            content=ft.Text('Задать пористость'),
                                            on_click=lambda e: page.open(porosity_changer)
                                        ),
                                        ft.Divider(),
                                        ft.MenuItemButton(
                                            content=ft.Text('Сбросить увеличение'),
                                            on_click=reset_viewer
                                        )
                                    ]
                                ),
                                ft.MenuItemButton(
                                    content=ft.Text('О приложении'),
                                    on_click=lambda e: page.open(about)
                                )
                            ]
                        )
                    ],
                    expand=True,
                    style=ft.MenuStyle(
                        alignment=ft.alignment.top_left,
                        bgcolor=COLOR_SCHEME['secondary'],
                        elevation=0,
                        side=ft.BorderSide(
                            width=0,
                            color=COLOR_SCHEME['secondary']
                        ),
                        mouse_cursor={
                            ft.ControlState.HOVERED: ft.MouseCursor.WAIT,
                            ft.ControlState.DEFAULT: ft.MouseCursor.ZOOM_OUT,
                        },
                    ),
                ),
                ft.Row(
                    [
                        ft.Text('')
                    ]
                ),
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.Icons.KEYBOARD_ARROW_LEFT,
                            icon_color=COLOR_SCHEME['primary'] + ',0.25',
                            disabled=True,
                            on_click=previous_image
                        ),
                        ft.Text(''),
                        ft.IconButton(
                            icon=ft.Icons.KEYBOARD_ARROW_RIGHT,
                            icon_color=COLOR_SCHEME['primary'],
                            on_click=next_image
                        ),
                        ft.IconButton(
                            icon=ft.Icons.PHOTO_LIBRARY,
                            icon_color=COLOR_SCHEME['primary'],
                            on_click=to_library
                        )
                    ],
                    expand=True,
                    alignment=ft.MainAxisAlignment.END
                )
            ]
        ),
        margin=ft.margin.all(-5),
        padding=ft.padding.only(right=5),
        bgcolor=COLOR_SCHEME['secondary']
    )
    # Status bar
    status_bar = ft.Container(
        ft.Row(
            [
                ft.Row(
                    [
                        ft.Text(value='', size=11, color='grey'),  # viewer scale
                        ft.Text(value='', size=11, color='grey'),  # image scale
                        ft.Text(value='', size=11, color='grey'),  # viewer offset
                        ft.Text('', size=11, color='grey')
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=30
                ),
                ft.Row(
                    [
                        ft.Text('|', size=11, color='grey'),
                        ft.Text('', size=11, color='grey'),
                        ft.Text('|', size=11, color='grey')
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True
                ),
                ft.Row(
                    [
                        ft.Text('', size=11, color='grey'),
                        ft.Text('', size=11, color='grey'),
                        ft.Text('', size=11, color='grey')
                    ],
                    alignment=ft.MainAxisAlignment.END,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=30
                )
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=COLOR_SCHEME['secondary'],
        margin=ft.margin.only(-10, -5, -10, -10),
        padding=ft.padding.only(15, 5, 15, 10)
    )
    # Toolbar
    toolbar = ft.NavigationRail(
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.AUTO_FIX_HIGH_OUTLINED,
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.ADD_CIRCLE_OUTLINE,
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.REMOVE_CIRCLE_OUTLINE,
            )
        ],
        width=50,
        min_width=56,
        label_type='none',
        bgcolor=COLOR_SCHEME['secondary'],
        indicator_shape=ft.CircleBorder(),
        indicator_color=COLOR_SCHEME['highlight'],
        on_change=change_tool
    )
    # Canvas
    canvas = ft.Stack(
        [
            ft.Stack(
                [
                    ft.Row(
                        [ft.Text('Чтобы открыть изображение, нажмите "Файл" -> "Открыть"')],
                        alignment=ft.MainAxisAlignment.CENTER
                    )
                ],
                alignment=ft.alignment.center_right,
                expand=True
            )
        ],
        alignment=ft.alignment.center_left,
        expand=True
    )

    # Initialize image
    open_image(page.session.get('sample_id'), page.session.get('image_index'))

    # Initialize editor view
    view = ft.View(
        '/library',
        controls=[
            ft.Column(
                [
                    menubar,
                    canvas,
                    status_bar
                ],
                expand=True
            )
        ],
        bgcolor=COLOR_SCHEME['background'],
        padding=ft.padding.all(5)
    )

    return view
