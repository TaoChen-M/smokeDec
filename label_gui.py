"""Label GUI"""
# coding: utf-8
# pylint: disable=no-member

import sys
import os
import threading
import time
from tkinter import messagebox, filedialog
from tkinter.simpledialog import askinteger
import tkinter as tk
from xml.dom import minidom
import numpy as np
from PIL import Image, ImageTk
import cv2

from detectors import Parallel, Skeleton, Hybrid
import configs


class LabelGUI(tk.Tk):
    """Label GUI"""
    image_width, image_height = 800, 640
    patch_width, patch_height = 400, 400
    title_name = 'Label'

    load_button_name = 'Load Image'
    reset_button_name = 'Reset'
    export_button_name = 'Export Patch Images'
    save_button_name = 'Save Results'
    exit_button_name = 'Exit'

    detect_button_name = 'Detect Lines'
    import_button_name = 'Import XML File'

    def __init__(self):
        super().__init__()
        self.title(self.title_name)
        self.resizable(False, False)
        self.grid_rowconfigure(0, minsize=LabelGUI.image_height + 80)
        self.grid_columnconfigure(0, minsize=LabelGUI.image_width + 20)
        self.grid_columnconfigure(1, minsize=LabelGUI.patch_width + 50)
        self.protocol('WM_DELETE_WINDOW', self._on_exit)

        cfg = configs.init_config()
        np.random.seed(0)
        if cfg.detector.method == 'parallel':
            self.detector = Parallel()
        elif cfg.detector.method == 'skeleton':
            self.detector = Skeleton()
        elif cfg.detector.method == 'hybrid':
            self.detector = Hybrid()
        elif cfg.detector.method == 'duo':
            self.detector = Parallel()
            self.detector_1 = Skeleton()
        else:
            return

        self.img, self.img_mask, self.img_patch, self.img_original = None, None, None, None
        self.img_name = None
        self.tk_img, self.tk_patch = None, None
        self.curr_is_original = True

        self.img_display, self.img_mask_display = None, None
        self.patch_index = None
        self.lines, self.patches = None, None
        self.states = None
        self.num_checked = None

        # Main menu
        self.main_menu = tk.Menu()
        self.config(menu=self.main_menu)

        # File menu
        self.file_menu = tk.Menu(tearoff=False)
        self.main_menu.add_cascade(label='File', menu=self.file_menu)

        self.file_menu.add_command(label=self.load_button_name, command=self._on_load)
        self.file_menu.add_command(label=self.reset_button_name, command=self._on_reset)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=self.export_button_name, command=self._on_export)
        self.file_menu.add_command(label=self.save_button_name, command=self._on_save)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=self.exit_button_name, command=self._on_exit)

        # Run menu
        self.run_menu = tk.Menu(tearoff=False)
        self.main_menu.add_cascade(label='Run', menu=self.run_menu)

        self.run_menu.add_command(label=self.detect_button_name, command=self._on_detect)
        self.run_menu.add_separator()
        self.run_menu.add_command(label=self.import_button_name, command=self._on_import)

        # Frame of image
        frame_1 = tk.Frame(self)
        frame_1.grid(row=0, column=0)

        # Show image
        self.image_canvas = tk.Canvas(frame_1, width=LabelGUI.image_width,
                                      height=LabelGUI.image_height, bg='#E0E0E0')
        self.image_canvas.pack()
        self.add_gap(frame_1, 14)

        # Switch button
        self.switch_button = tk.Button(frame_1, text="Switch", width=16,
                                       command=self._on_switch)
        self.switch_button.pack(side='left', expand=True)

        # Frame of patch
        frame_2 = tk.Frame(self)
        frame_2.grid(row=0, column=1)

        # Show patch
        self.patch_canvas = tk.Canvas(frame_2, width=LabelGUI.patch_width,
                                      height=LabelGUI.patch_height, bg='#E0E0E0')
        self.patch_canvas.pack()
        self.add_gap(frame_2, 12)

        frame_3 = tk.Frame(frame_2)
        frame_3.pack()
        self.add_gap(frame_2, 12)

        # Previous button
        self.previous_button = tk.Button(frame_3, text="Previous", width=8,
                                         command=self._on_previous)
        self.previous_button.pack(side='left')

        # Jump button
        self.jump_button = tk.Button(frame_3, text="Jump", width=16,
                                     command=self._on_jump)
        self.jump_button.pack(side='left')

        # Next button
        self.next_button = tk.Button(frame_3, text="Next", width=8,
                                     command=self._on_next)
        self.next_button.pack(side='left')

        # Confirm button
        self.confirm_button = tk.Button(frame_2, text="Confirm", width=16,
                                        command=self._on_confirm)
        self.confirm_button.pack()
        self.add_gap(frame_2, 12)

        # Remove button
        self.remove_button = tk.Button(frame_2, text="Remove", width=16,
                                       command=self._on_remove)
        self.remove_button.pack()
        self.add_gap(frame_2, 12)

        self.file_menu.entryconfig(self.reset_button_name, state='disabled')
        self.file_menu.entryconfig(self.export_button_name, state='disabled')
        self.file_menu.entryconfig(self.save_button_name, state='disabled')
        self.run_menu.entryconfig(self.detect_button_name, state='disabled')
        self.run_menu.entryconfig(self.import_button_name, state='disabled')
        self.switch_button.config(state='disabled')
        self.previous_button.config(state='disabled')
        self.jump_button.config(state='disabled')
        self.next_button.config(state='disabled')
        self.confirm_button.config(state='disabled')
        self.remove_button.config(state='disabled')

    def _on_load(self):
        """Click load button"""
        img_path = filedialog.askopenfilename(
            title='Load Image',
            filetypes=[('Image Files', '*.jpg *.png'),
                       ('All Files', '*')],
            initialdir=os.getcwd())
        if not img_path or not os.path.exists(img_path):
            return

        self._on_reset()

        self.img_name = ''.join((os.path.split(img_path)[-1]).split('.')[:-1])
        self.img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.img_original = self.img.copy()

        self._draw_canvas(self.img, 'image')

        self.file_menu.entryconfig(self.reset_button_name, state='active')
        self.run_menu.entryconfig(self.detect_button_name, state='active')
        self.run_menu.entryconfig(self.import_button_name, state='active')

    def _on_detect(self):
        """Click detect button"""
        threading.Thread(target=self._get_detected_results, daemon=True).start()
        time.sleep(0.1)

    def _on_import(self):
        """Click import button"""
        xml_path = filedialog.askopenfilename(
            title='Load xml',
            filetypes=[('XML Files', '*.xml')],
            initialdir=os.getcwd())
        if not xml_path or not os.path.exists(xml_path):
            return

        threading.Thread(target=self._get_detected_results, args=(xml_path,)).start()
        time.sleep(0.1)

    def _on_switch(self):
        """Click switch button"""
        if self.curr_is_original:
            self._draw_canvas(self.img_mask_display, 'image')
            self.curr_is_original = False
        else:
            self._draw_canvas(self.img_display, 'image')
            self.curr_is_original = True

    def _on_reset(self):
        """Click reset button"""
        if self.img is None or \
                not messagebox.askokcancel('Reset', 'Are you sure you want to reset?'):
            return
        self.title(self.title_name)
        self.img, self.img_mask, self.img_patch = None, None, None
        self.img_name = None
        self.tk_img, self.tk_patch = None, None
        self.curr_is_original = True
        self.img_display, self.img_mask_display = None, None
        self.patch_index = None
        self.lines, self.patches = None, None
        self.states = None
        self.num_checked = None

        self.file_menu.entryconfig(self.reset_button_name, state='disabled')
        self.file_menu.entryconfig(self.export_button_name, state='disabled')
        self.file_menu.entryconfig(self.save_button_name, state='disabled')
        self.run_menu.entryconfig(self.detect_button_name, state='disabled')
        self.run_menu.entryconfig(self.import_button_name, state='disabled')
        self.switch_button.config(state='disabled')
        self.previous_button.config(state='disabled')
        self.jump_button.config(state='disabled')
        self.next_button.config(state='disabled')
        self.confirm_button.config(state='disabled')
        self.remove_button.config(state='disabled')

    def _on_previous(self):
        """Click previous button"""
        self.patch_index -= 1
        if self.patch_index < 0:
            self.patch_index = len(self.patches) - 1
        self.title(self.title_name + ' - (current: ' + str(self.patch_index + 1) +
                   ', checked: ' + str(self.num_checked) +
                   ', total: ' + str(len(self.patches)) + ')')
        self._update_image_display()

    def _on_jump(self):
        """Click jump button"""
        i = askinteger('Jump', 'Please enter the number you want to jump to:',
                       minvalue=1, maxvalue=len(self.patches))
        if i is None:
            return
        self.patch_index = i - 1
        self.title(self.title_name + ' - (current: ' + str(self.patch_index + 1) +
                   ', checked: ' + str(self.num_checked) +
                   ', total: ' + str(len(self.patches)) + ')')
        self._update_image_display()

    def _on_next(self):
        """Click next button"""
        self.patch_index += 1
        if self.patch_index >= len(self.patches):
            self.patch_index = 0

        self.title(self.title_name + ' - (current: ' + str(self.patch_index + 1) +
                   ', checked: ' + str(self.num_checked) +
                   ', total: ' + str(len(self.patches)) + ')')
        self._update_image_display()

    def _on_confirm(self):
        """Click confirm button"""
        if self.states[self.patch_index] == 2:
            self.num_checked += 1
        self.states[self.patch_index] = 1

        curr_line = self.lines[self.patch_index]
        point_1, point_2 = (curr_line[0], curr_line[1]), (curr_line[2], curr_line[3])
        cv2.line(self.img, point_1, point_2, (0, 255, 0), 1)
        cv2.line(self.img_mask, point_1, point_2, (0, 255, 0), 1)

        self.patch_index += 1
        if self.patch_index >= len(self.patches):
            self.patch_index = 0
        self.title(self.title_name + ' - (current: ' + str(self.patch_index + 1) +
                   ', checked: ' + str(self.num_checked) +
                   ', total: ' + str(len(self.patches)) + ')')
        self._update_image_display()

    def _on_remove(self):
        """Click remove button"""
        if self.states[self.patch_index] == 2:
            self.num_checked += 1
        self.states[self.patch_index] = 0

        curr_line = self.lines[self.patch_index]
        point_1, point_2 = (curr_line[0], curr_line[1]), (curr_line[2], curr_line[3])
        cv2.line(self.img, point_1, point_2, (0, 0, 255), 1)
        cv2.line(self.img_mask, point_1, point_2, (0, 0, 255), 1)

        self.patch_index += 1
        if self.patch_index >= len(self.patches):
            self.patch_index = 0
        self.title(self.title_name + ' - (current: ' + str(self.patch_index + 1) +
                   ', checked: ' + str(self.num_checked) +
                   ', total: ' + str(len(self.patches)) + ')')
        self._update_image_display()

    def _on_save(self):
        """Click save button"""
        save_path = filedialog.asksaveasfilename(
            title='Save results',
            filetypes=[('XML Files', '*.xml')],
            initialdir=os.getcwd(),
            initialfile=self.img_name + time.strftime(' %Y_%m_%d %H_%M_%S') + '.xml')
        if not save_path:
            return
        self._write_xml(save_path)
        messagebox.showinfo('Info', 'Successfully saved')

    def _on_export(self):
        """Click export button"""
        export_dir = filedialog.askdirectory(
            title='Export patch images',
            initialdir=os.getcwd())
        if not export_dir:
            return
        export_dir = os.path.join(export_dir,
                                  'patches ' + self.img_name + time.strftime(' %Y_%m_%d %H_%M_%S'))
        self._export_patches(export_dir)
        messagebox.showinfo('Info', 'Successfully exported')

    def _on_empty_patches(self):
        """When patches are empty"""
        self.tk_patch = None
        self._draw_canvas(self.img, 'image')
        self.switch_button.config(state='disabled')
        self.previous_button.config(state='disabled')
        self.jump_button.config(state='disabled')
        self.next_button.config(state='disabled')
        self.confirm_button.config(state='disabled')
        self.remove_button.config(state='disabled')

    def _on_exit(self):
        """Click exit"""
        if messagebox.askokcancel('Confirm Exit', 'Are you sure you want to exit?'):
            sys.exit(0)

    def _get_detected_results(self, xml_path=None):
        """Start detect thread"""
        self.file_menu.entryconfig(self.load_button_name, state='disabled')
        self.file_menu.entryconfig(self.reset_button_name, state='disabled')
        self.run_menu.entryconfig(self.detect_button_name, state='disabled')
        self.run_menu.entryconfig(self.import_button_name, state='disabled')

        self.img_mask, binary, contours = self.detector.preprocess(self.img)

        if xml_path is None:
            self.title(self.title_name + ' - detecting')
            self.lines, self.patches = self.detector.run(binary, contours)
            self.states = [2] * len(self.patches)
            self.num_checked = 0
        else:
            self.title(self.title_name + ' - reading')
            self._read_xml(xml_path)

        if len(self.lines) == 0 or len(self.patches) == 0:
            self._on_empty_patches()
            return

        self.patch_index = 0
        self.curr_is_original = False
        self._update_image_display()

        self.title(self.title_name + ' - (current: ' + str(self.patch_index + 1) +
                   ', checked: ' + str(self.num_checked) +
                   ', total: ' + str(len(self.patches)) + ')')
        self.file_menu.entryconfig(self.load_button_name, state='active')
        self.file_menu.entryconfig(self.reset_button_name, state='active')
        self.file_menu.entryconfig(self.export_button_name, state='active')
        self.file_menu.entryconfig(self.save_button_name, state='active')
        self.switch_button.config(state='active')
        self.previous_button.config(state='active')
        self.jump_button.config(state='active')
        self.next_button.config(state='active')
        self.confirm_button.config(state='active')
        self.remove_button.config(state='active')

    def _update_image_display(self):
        """Update images"""
        self.img_display = self.img.copy()
        self.img_mask_display = self.img_mask.copy()
        curr_patch, curr_line = self.patches[self.patch_index], self.lines[self.patch_index]
        top_left, bottom_right = (curr_patch[2], curr_patch[0]), (curr_patch[3], curr_patch[1])
        point_1, point_2 = (curr_line[0], curr_line[1]), (curr_line[2], curr_line[3])

        cv2.rectangle(self.img_display, top_left, bottom_right, (0, 0, 255), 5)
        cv2.rectangle(self.img_mask_display, top_left, bottom_right, (0, 0, 255), 5)
        cv2.line(self.img_display, point_1, point_2, (255, 255, 0), 1)
        cv2.line(self.img_mask_display, point_1, point_2, (255, 255, 0), 1)

        self.img_patch = \
            self.img_original[curr_patch[0]: curr_patch[1], curr_patch[2]: curr_patch[3]].copy()
        point_1 = (point_1[0] - curr_patch[2], point_1[1] - curr_patch[0])
        point_2 = (point_2[0] - curr_patch[2], point_2[1] - curr_patch[0])
        if self.states[self.patch_index] == 1:
            cv2.line(self.img_patch, point_1, point_2, (0, 255, 0), 1)
        elif self.states[self.patch_index] == 0:
            cv2.line(self.img_patch, point_1, point_2, (0, 0, 255), 1)
        else:
            cv2.line(self.img_patch, point_1, point_2, (255, 255, 0), 1)
        if self.curr_is_original:
            self._draw_canvas(self.img_display, 'image')
        else:
            self._draw_canvas(self.img_mask_display, 'image')
        self._draw_canvas(self.img_patch, 'patch')

    def _draw_canvas(self, img, target):
        """Draw the image on the specific canvas"""
        if target == 'image':
            canvas = self.image_canvas
            width, height = self.image_width, self.image_height
        elif target == 'patch':
            canvas = self.patch_canvas
            width, height = self.patch_width, self.patch_height
        else:
            raise ValueError('Wrong target')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((width, height))

        canvas.delete('all')
        image_item = canvas.create_image(0, 0, anchor='nw')

        if target == 'image':
            self.tk_img = ImageTk.PhotoImage(img)
            canvas.itemconfig(image_item, image=self.tk_img)
        else:
            self.tk_patch = ImageTk.PhotoImage(img)
            canvas.itemconfig(image_item, image=self.tk_patch)

    def _read_xml(self, path):
        """Read lines and patches from xml"""
        self.lines, self.patches, self.states = [], [], []
        self.num_checked = 0

        dom = minidom.parse(path)
        root_node = dom.documentElement
        for target_node in root_node.getElementsByTagName('target'):
            line_node = target_node.getElementsByTagName('line')[0]
            x0 = int(line_node.getElementsByTagName('x0')[0].firstChild.data)
            y0 = int(line_node.getElementsByTagName('y0')[0].firstChild.data)
            x1 = int(line_node.getElementsByTagName('x1')[0].firstChild.data)
            y1 = int(line_node.getElementsByTagName('y1')[0].firstChild.data)
            self.lines.append((x0, y0, x1, y1))

            patch_node = target_node.getElementsByTagName('patch')[0]
            top = int(patch_node.getElementsByTagName('top')[0].firstChild.data)
            bottom = int(patch_node.getElementsByTagName('bottom')[0].firstChild.data)
            left = int(patch_node.getElementsByTagName('left')[0].firstChild.data)
            right = int(patch_node.getElementsByTagName('right')[0].firstChild.data)
            self.patches.append((top, bottom, left, right))

            state_node = target_node.getElementsByTagName('state')[0]
            state = state_node.firstChild.data
            if state == 'negative':
                self.states.append(0)
                self.num_checked += 1
            elif state == 'positive':
                self.states.append(1)
                self.num_checked += 1
            else:
                self.states.append(2)

    def _write_xml(self, path):
        """Save target info as xml"""
        dom = minidom.Document()
        root_node = dom.createElement('results')
        dom.appendChild(root_node)
        for idx in range(len(self.patches)):
            target_node = dom.createElement('target')
            root_node.appendChild(target_node)

            line_node = dom.createElement('line')
            target_node.appendChild(line_node)

            for i, name in enumerate(['x0', 'y0', 'x1', 'y1']):
                curr_node = dom.createElement(name)
                curr_node.appendChild(dom.createTextNode(str(self.lines[idx][i])))
                line_node.appendChild(curr_node)

            patch_node = dom.createElement('patch')
            target_node.appendChild(patch_node)
            for i, name in enumerate(['top', 'bottom', 'left', 'right']):
                curr_node = dom.createElement(name)
                curr_node.appendChild(dom.createTextNode(str(self.patches[idx][i])))
                patch_node.appendChild(curr_node)

            if self.states[idx] == 0:
                state = 'negative'
            elif self.states[idx] == 1:
                state = 'positive'
            else:
                state = 'unknown'

            state_node = dom.createElement('state')
            state_node.appendChild(dom.createTextNode(state))
            target_node.appendChild(state_node)

        with open(path, 'w') as file:
            file.write(dom.toprettyxml())

    def _export_patches(self, export_dir):
        """Export patch images"""
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        for idx, line, patch, state in \
                zip(range(len(self.lines)), self.lines, self.patches, self.states):
            top, bottom, left, right = patch
            patch_image = self.img_original[top:bottom, left:right].copy()
            cv2.line(patch_image, (line[0] - left, line[1] - top), (line[2] - left, line[3] - top),
                     (0, 0, 255))

            save_path = os.path.join(export_dir, str(idx) + '_' + str(state) + '.png')
            cv2.imencode('.png', patch_image)[1].tofile(save_path)

    @staticmethod
    def add_gap(master, height, side='top'):
        """Add gap"""
        tk.Frame(master, height=height).pack(side=side)


if __name__ == '__main__':
    LabelGUI().mainloop()
