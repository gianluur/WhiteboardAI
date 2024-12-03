"""
MIT License

Copyright (c) 2024 Gianluca Russo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
from typing import Tuple, Text
from dataclasses import dataclass


@dataclass
class DrawingState:
	"""Maintains the state of drawing parameters"""
	canvas: np.ndarray
	color: Tuple[int, int, int]
	thickness: int
	index_prev_x: int | None
	index_prev_y: int | None
	sensitivity: int


class ColorPalette:
	"""Manages color selection and drawing color boxes"""
	WHITE = (255, 255, 255)
	BLACK = (0, 0, 0)
	RED = (0, 0, 255)
	GREEN = (0, 255, 0)
	BLUE = (255, 0, 0)
	
	COLOR_BOXES = [
		{"color": RED, "pos": (350, 2, 400, 52), "name": "Red"},
		{"color": GREEN, "pos": (410, 2, 460, 52), "name": "Green"},
		{"color": BLUE, "pos": (470, 2, 520, 52), "name": "Blue"},
		{"color": BLACK, "pos": (530, 2, 580, 52), "name": "Black"},
		{"color": WHITE, "pos": (590, 2, 640, 52), "name": "White"}
	]
	
	@staticmethod
	def draw_color_boxes(frame: np.ndarray) -> None:
		"""Draw color selection boxes on the frame"""
		for box in ColorPalette.COLOR_BOXES:
			x1, y1, x2, y2 = box["pos"]
			color = box["color"]
			cv.rectangle(frame, (x1, y1), (x2, y2), color, -1)
			
			name = box["name"]
			
			font = cv.FONT_HERSHEY_SIMPLEX
			font_scale = 0.5
			font_thickness = 2
			text_color = (255, 255, 255) if color != ColorPalette.WHITE else (0, 0, 0)  # Make text black on white box
			
			# Calculate text size and position
			text_size = cv.getTextSize(name, font, font_scale, font_thickness)[0]
			text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center text horizontally
			text_y = y1 + (y2 - y1 + text_size[1]) // 2  # Center text vertically
			
			# Put the text in the box
			cv.putText(frame, name, (text_x, text_y), font, font_scale, text_color, font_thickness)
	
	@staticmethod
	def get_selected_color(x: int, y: int) -> Tuple[int, int, int] | None:
		"""Determine if a color is selected based on coordinates"""
		for box in ColorPalette.COLOR_BOXES:
			x1, y1, x2, y2 = box["pos"]
			if x <= x2 and y1 <= y <= y2:
				return box["color"]
		return None


class DrawingTools:
	TOOLS = [
		{"name": "Clear", "pos": (720, 2, 770, 52), "color": ColorPalette.WHITE},
		{"name": "+", "pos": (780, 2, 830, 52), "color": ColorPalette.WHITE},
		{"name": "-", "pos": (840, 2, 890, 52), "color": ColorPalette.WHITE}
	]
	
	@staticmethod
	def draw_tool_boxes(frame: np.ndarray) -> None:
		"""Draw tools boxes on the frame"""
		for box in DrawingTools.TOOLS:
			x1, y1, x2, y2 = box["pos"]
			color = box["color"]
			cv.rectangle(frame, (x1, y1), (x2, y2), color, -1)
			
			name = box["name"]
			
			font = cv.FONT_HERSHEY_SIMPLEX
			font_scale = 0.5
			font_thickness = 2
			text_color = (255, 255, 255) if color != ColorPalette.WHITE else (0, 0, 0)
			
			# Calculate text size and position
			text_size = cv.getTextSize(name, font, font_scale, font_thickness)[0]
			text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center text horizontally
			text_y = y1 + (y2 - y1 + text_size[1]) // 2  # Center text vertically
			
			# Put the text in the box
			cv.putText(frame, name, (text_x, text_y), font, font_scale, text_color, font_thickness)
	
	@staticmethod
	def get_selected_tool(x: int, y: int) -> Text | None:
		"""Determine if a color is selected based on coordinates"""
		for box in DrawingTools.TOOLS:
			x1, y1, x2, y2 = box["pos"]
			if x <= x2 and y1 <= y <= y2:
				return box["name"]
		return None
	
	@staticmethod
	def clear(canvas: np.ndarray):
		canvas[:, :] = 255
	
	@staticmethod
	def increase_thickness(thickness):
		"""Increase thickness and return the new thickness"""
		return thickness + 1
	
	@staticmethod
	def decrease_thickness(thickness):
		"""Decrease thickness and return the new thickness"""
		return max(1, thickness - 1)


@dataclass
class Camera:
	"""Maintains the information about the camera"""
	window_name: Text
	width: int
	height: int


@dataclass
class Cursor:
	pos_x: int
	pos_y: int


class HandDetector:
	"""Handles hand detection and finger tracking"""
	
	def __init__(self):
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=False,
			max_num_hands=1,
			min_detection_confidence=0.5
		)
		self.mp_drawing = mp.solutions.drawing_utils
	
	def is_finger_up(self, landmarks, finger_tip, finger_dip, finger_pip) -> bool:
		"""Check if a finger is pointing upward"""
		if finger_pip == self.mp_hands.HandLandmark.THUMB_IP:
			return False
		return (landmarks[finger_tip].y < landmarks[finger_pip].y and
						landmarks[finger_tip].y < landmarks[finger_dip].y)
	
	@staticmethod
	def calculate_coordinates(finger_tip, frame) -> Tuple[int, int]:
		"""Calculate x,y coordinates for a fingertip"""
		frame_h, frame_w, _ = frame.shape
		x = int(finger_tip.x * frame_w)
		y = int(finger_tip.y * frame_h)
		return x, y


class HandDrawingApp:
	"""Main application class for hand drawing"""
	
	def __init__(self):
		self.detector = HandDetector()
		self.state = DrawingState(
			canvas=np.ones((450, 800, 3), np.uint8) * 255,
			color=ColorPalette.BLACK,
			thickness=4,
			index_prev_x=None,
			index_prev_y=None,
			sensitivity=20
		
		)
		self.camera = Camera(
			window_name='Webcam',
			width=1000,
			height=700
		)
		self.setup_camera()
		
		self.cursor = Cursor(
			pos_x=0,
			pos_y=0
		)
		self.setup_mouse_callback()
	
	def setup_camera(self):
		"""Initialize the camera with desired settings"""
		self.cam = cv.VideoCapture(0)
		self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self.camera.width)
		self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.camera.height)
	
	def setup_mouse_callback(self):
		"""Initialize mouse callback for tracking cursor position"""
		cv.namedWindow(self.camera.window_name)  # Make sure the window is named
		cv.setMouseCallback(self.camera.window_name, self.mouse_callback)
	
	def mouse_callback(self, event, x, y, flags, param):
		"""Capture mouse cursor position"""
		if event == cv.EVENT_MOUSEMOVE:
			self.cursor.pos_x, self.cursor.pos_y = x, y
	
	def process_frame(self, frame: np.ndarray):
		"""Process each frame for hand detection and drawing"""
		frame = cv.flip(frame, 1)  # 1 --> flips horizontally
		rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		ColorPalette.draw_color_boxes(frame)
		DrawingTools.draw_tool_boxes(frame)
		
		result = self.detector.hands.process(rgb_frame)
		
		if result.multi_hand_landmarks:
			self.process_hand_landmarks(frame, result)
		
		print(f"Cursor -> x: {self.cursor.pos_x} | y: {self.cursor.pos_y}")
		cv.putText(frame, f"Thickness: {self.state.thickness}", (2, 35), cv.FONT_HERSHEY_SIMPLEX, 0.85, ColorPalette.WHITE,
							 2)
		return frame
	
	def process_hand_landmarks(self, frame, result):
		"""Process detected hand landmarks for drawing"""
		for hand_landmarks, handedness in zip(result.multi_hand_landmarks,
																					result.multi_handedness):
			self.detector.mp_drawing.draw_landmarks(
				frame, hand_landmarks,
				self.detector.mp_hands.HAND_CONNECTIONS
			)
			
			landmarks = hand_landmarks.landmark
			self.handle_drawing(frame, landmarks)
	
	def handle_drawing(self, frame, landmarks):
		"""Handle the drawing logic based on finger positions"""
		is_index_up = self.detector.is_finger_up(
			landmarks,
			self.detector.mp_hands.HandLandmark.INDEX_FINGER_TIP,
			self.detector.mp_hands.HandLandmark.INDEX_FINGER_DIP,
			self.detector.mp_hands.HandLandmark.INDEX_FINGER_PIP
		)
		
		is_pinky_up = self.detector.is_finger_up(
			landmarks,
			self.detector.mp_hands.HandLandmark.PINKY_TIP,
			self.detector.mp_hands.HandLandmark.PINKY_DIP,
			self.detector.mp_hands.HandLandmark.PINKY_PIP
		)
		
		self.handle_index_drawing(frame, landmarks, is_index_up)
		self.handle_color_selection(frame, landmarks, is_pinky_up)
	
	def handle_index_drawing(self, frame, landmarks, is_index_up):
		"""Handle drawing with index finger"""
		if is_index_up:
			index_tip = landmarks[self.detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
			x, y = self.detector.calculate_coordinates(index_tip, frame)
			cv.circle(frame, (x, y), 15, self.state.color, -1)
			
			x += self.state.sensitivity
			y += self.state.sensitivity
			if self.state.index_prev_x is not None:
				cv.line(
					self.state.canvas,
					(self.state.index_prev_x, self.state.index_prev_y),
					(x, y),
					self.state.color,
					self.state.thickness
				)
			self.state.index_prev_x, self.state.index_prev_y = x, y
		else:
			self.state.index_prev_x = self.state.index_prev_y = None
	
	def handle_color_selection(self, frame, landmarks, is_pinky_up):
		"""Handle color selection with pinky finger"""
		if is_pinky_up:
			pinky_tip = landmarks[self.detector.mp_hands.HandLandmark.PINKY_TIP]
			x, y = self.detector.calculate_coordinates(pinky_tip, frame)
			cv.circle(frame, (x, y), 15, self.state.color, -1)
			
			new_color = ColorPalette.get_selected_color(x, y)
			if new_color:
				self.state.color = new_color
			else:
				new_tool = DrawingTools.get_selected_tool(x, y)
				if new_tool == 'Clear':
					DrawingTools.clear(self.state.canvas)
				
				if new_tool == '+':
					self.state.thickness = DrawingTools.increase_thickness(self.state.thickness)
				
				if new_tool == '-':
					self.state.thickness = DrawingTools.decrease_thickness(self.state.thickness)
	
	def run(self):
		"""Main application loop"""
		while self.cam.isOpened():
			success, frame = self.cam.read()
			if not success:
				break
			
			processed_frame = self.process_frame(frame)
			
			cv.imshow(self.camera.window_name, processed_frame)
			cv.imshow('Canvas', self.state.canvas)
			
			if cv.waitKey(1) == 27:  # ESC key
				break
		
		self.cleanup()
	
	def cleanup(self):
		"""Clean up resources"""
		self.cam.release()
		cv.destroyAllWindows()


def main():
	app = HandDrawingApp()
	app.run()


if __name__ == "__main__":
	main()