# sketch_corrector.py
# --- MODIFIED AND IMPROVED VERSION ---

import torch
import numpy as np
import cv2
import os
import math

class SketchCorrector:
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        # --- ADDED PENTAGON AND HEXAGON TO THE LIST ---
        self.shapes = ['circle', 'square', 'rectangle', 'triangle', 'pentagon', 'hexagon', 'diamond', 'star', 'heart', 'oval']
        self.initialize_models()

    def initialize_models(self):
        # Initialize models for different shapes
        self.models = {shape: None for shape in self.shapes}

    def detect_shape(self, stroke):
        """
        --- REWRITTEN FOR BETTER ACCURACY ---
        Detect the most likely shape from the stroke using a simplified and more robust method.
        """
        if len(stroke) < 5: # Need at least a few points to detect a shape
            return 'unknown'

        # Convert stroke to a numpy array for OpenCV
        stroke_np = np.array(stroke, dtype=np.int32)
        
        # Create a contour from the stroke points
        contour = stroke_np.reshape(-1, 1, 2)
        
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 100: # Ignore very small drawings
             return 'unknown'

        # Approximate the contour to a polygon. The epsilon value is key.
        # A value between 0.03 and 0.05 allows for more tolerance in hand-drawing.
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)

        # Get the bounding box to help differentiate quadrilaterals and ellipses
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            return 'unknown'

        # Identify shape based on the number of vertices
        if num_vertices == 3:
            return 'triangle'
        
        elif num_vertices == 4:
            # For 4 vertices, it can be a square, rectangle, or diamond.
            aspect_ratio = w / float(h)
            if 0.85 <= aspect_ratio <= 1.15:
                return 'square'
            else:
                return 'rectangle'
        
        elif num_vertices == 5:
            return 'pentagon'
            
        elif num_vertices == 6:
            return 'hexagon'

        # If it has many vertices, it's likely a curved shape like a circle or oval.
        elif num_vertices > 6:
            if len(contour) < 5:
                return 'unknown'

            try:
                (cx, cy), (minor_axis, major_axis), angle = cv2.fitEllipse(contour)
                if major_axis == 0: return 'unknown'
                axis_ratio = minor_axis / float(major_axis)

                if axis_ratio > 0.8:
                    return 'circle'
                else:
                    return 'oval'
            except cv2.error:
                # Fallback if ellipse fitting fails, check aspect ratio
                aspect_ratio = w / float(h)
                return 'circle' if 0.8 <= aspect_ratio <= 1.2 else 'oval'
        
        return 'unknown'

    def correct_stroke(self, stroke, shape=None):
        if not stroke or len(stroke) < 3:
            return stroke

        stroke = np.array(stroke)
        
        if shape is None:
            shape = self.detect_shape(stroke)
        
        if shape == 'circle':
            return self._correct_circle(stroke)
        elif shape == 'square':
            return self._correct_square(stroke)
        elif shape == 'rectangle':
            return self._correct_rectangle(stroke)
        elif shape == 'triangle':
            return self._correct_triangle(stroke)
        # --- ADDED CALLS FOR NEW SHAPES ---
        elif shape == 'pentagon':
            return self._correct_pentagon(stroke)
        elif shape == 'hexagon':
            return self._correct_hexagon(stroke)
        elif shape == 'diamond':
            return self._correct_diamond(stroke)
        elif shape == 'star':
            return self._correct_star(stroke)
        elif shape == 'heart':
            return self._correct_heart(stroke)
        elif shape == 'oval':
            return self._correct_oval(stroke)
        else:
            return stroke

    # --- NEW FUNCTION TO DRAW A PENTAGON ---
    def _correct_pentagon(self, stroke):
        """Correct stroke to a perfect pentagon."""
        center = np.mean(stroke, axis=0)
        distances = np.linalg.norm(stroke - center, axis=1)
        radius = np.mean(distances)
        
        points = []
        n_points = len(stroke)
        points_per_side = max(1, n_points // 5)
        
        # Generate 5 vertices of a regular pentagon
        vertices = []
        for i in range(5):
            angle = 2 * np.pi * i / 5 - (np.pi / 2) # Start from the top
            vx = center[0] + radius * np.cos(angle)
            vy = center[1] + radius * np.sin(angle)
            vertices.append(np.array([vx, vy]))
            
        # Create points along the edges
        for i in range(5):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 5]
            for j in range(points_per_side):
                t = j / float(points_per_side)
                point = start_vertex * (1 - t) + end_vertex * t
                points.append(point)

        return np.array(points).astype(int)

    # --- NEW FUNCTION TO DRAW A HEXAGON ---
    def _correct_hexagon(self, stroke):
        """Correct stroke to a perfect hexagon."""
        center = np.mean(stroke, axis=0)
        distances = np.linalg.norm(stroke - center, axis=1)
        radius = np.mean(distances)

        points = []
        n_points = len(stroke)
        points_per_side = max(1, n_points // 6)

        # Generate 6 vertices of a regular hexagon
        vertices = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            vx = center[0] + radius * np.cos(angle)
            vy = center[1] + radius * np.sin(angle)
            vertices.append(np.array([vx, vy]))

        # Create points along the edges
        for i in range(6):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 6]
            for j in range(points_per_side):
                t = j / float(points_per_side)
                point = start_vertex * (1 - t) + end_vertex * t
                points.append(point)

        return np.array(points).astype(int)
        
    def _correct_circle(self, stroke):
        center = np.mean(stroke, axis=0)
        radius = np.mean(np.linalg.norm(stroke - center, axis=1))
        angles = np.linspace(0, 2 * np.pi, len(stroke))
        corrected = np.array([
            center + radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ])
        return corrected.astype(int)

    def _correct_square(self, stroke):
        x, y, w, h = cv2.boundingRect(np.array(stroke))
        center_x, center_y = x + w / 2, y + h / 2
        size = max(w, h)
        
        min_x, max_x = center_x - size / 2, center_x + size / 2
        min_y, max_y = center_y - size / 2, center_y + size / 2
        
        vertices = [np.array([min_x, min_y]), np.array([max_x, min_y]), np.array([max_x, max_y]), np.array([min_x, max_y])]
        
        return self._generate_points_from_vertices(vertices, len(stroke))

    def _correct_rectangle(self, stroke):
        x, y, w, h = cv2.boundingRect(np.array(stroke))
        min_x, max_x = x, x + w
        min_y, max_y = y, y + h

        vertices = [np.array([min_x, min_y]), np.array([max_x, min_y]), np.array([max_x, max_y]), np.array([min_x, max_y])]
        
        return self._generate_points_from_vertices(vertices, len(stroke))

    def _correct_triangle(self, stroke):
        x, y, w, h = cv2.boundingRect(np.array(stroke))
        
        vertices = [np.array([x + w / 2, y]), np.array([x + w, y + h]), np.array([x, y + h])]
        
        return self._generate_points_from_vertices(vertices, len(stroke))
        
    def _correct_diamond(self, stroke):
        x, y, w, h = cv2.boundingRect(np.array(stroke))
        
        vertices = [np.array([x + w / 2, y]), np.array([x + w, y + h / 2]), np.array([x + w / 2, y + h]), np.array([x, y + h / 2])]
        
        return self._generate_points_from_vertices(vertices, len(stroke))

    def _correct_oval(self, stroke):
        x, y, w, h = cv2.boundingRect(np.array(stroke))
        center_x, center_y = x + w / 2, y + h / 2
        a, b = w / 2, h / 2
        
        points = []
        n_points = len(stroke)
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            px = center_x + a * np.cos(angle)
            py = center_y + b * np.sin(angle)
            points.append([px, py])
            
        return np.array(points).astype(int)
        
    def _correct_star(self, stroke):
        center = np.mean(stroke, axis=0)
        # Find outer radius by looking at points furthest from center
        distances = np.linalg.norm(stroke - center, axis=1)
        outer_radius = np.percentile(distances, 95) # Use 95th percentile to be robust to outliers
        inner_radius = outer_radius * 0.5
        
        num_spikes = 5
        vertices = []
        for i in range(num_spikes * 2):
            r = outer_radius if i % 2 == 0 else inner_radius
            angle = i * np.pi / num_spikes - (np.pi / 2)
            vertices.append(center + r * np.array([np.cos(angle), np.sin(angle)]))
            
        return self._generate_points_from_vertices(vertices, len(stroke))

    def _correct_heart(self, stroke):
        x, y, w, h = cv2.boundingRect(np.array(stroke))
        points = []
        n_points = len(stroke)

        t = np.linspace(0, 2*np.pi, n_points)
        px = x + w/2 + (w/2.1) * (16 * np.sin(t)**3) / 16
        py = y + h/2.1 - (h/2.3) * (13 * np.cos(t) - 5 * np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) / 16
        
        return np.array([px, py]).T.astype(int)

    def _generate_points_from_vertices(self, vertices, n_points):
        """Helper function to generate a stroke from a list of vertices."""
        points = []
        num_sides = len(vertices)
        points_per_side = max(1, n_points // num_sides)

        for i in range(num_sides):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % num_sides]
            for j in range(points_per_side):
                t = j / float(points_per_side)
                point = start_vertex * (1 - t) + end_vertex * t
                points.append(point)

        return np.array(points).astype(int)

    def get_available_shapes(self):
        return self.shapes