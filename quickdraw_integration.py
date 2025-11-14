import numpy as np
import tensorflow as tf
from quickdraw import QuickDrawData
import cv2
import os

class QuickDrawRecognizer:
    def __init__(self, categories=None):
        self.categories = categories or ['circle', 'square', 'triangle', 'star']
        self.model = None
        self.initialize_model()
        
    def initialize_model(self):
        # Load pre-trained model for sketch recognition
        model_path = 'models/quickdraw_model.h5'
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.train_model()
    
    def train_model(self):
        # Create and train a simple CNN model for sketch recognition
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.categories), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train the model using QuickDraw data
        for category in self.categories:
            qd = QuickDrawData()
            drawings = qd.get_drawing(category)
            # Process and train with the drawings
            # This is a simplified version - you'll need to implement proper data processing
        
        self.model = model
        model.save('models/quickdraw_model.h5')
    
    def preprocess_stroke(self, stroke):
        # Convert stroke to image format
        img = np.zeros((28, 28), dtype=np.uint8)
        for point in stroke:
            x, y = int(point[0] * 27), int(point[1] * 27)
            if 0 <= x < 28 and 0 <= y < 28:
                img[y, x] = 255
        return img.reshape(1, 28, 28, 1)
    
    def recognize_stroke(self, stroke):
        if self.model is None:
            return None
        
        # Preprocess the stroke
        processed_stroke = self.preprocess_stroke(stroke)
        
        # Get prediction
        prediction = self.model.predict(processed_stroke)
        category_idx = np.argmax(prediction)
        
        return {
            'category': self.categories[category_idx],
            'confidence': float(prediction[0][category_idx])
        }
    
    def get_suggestions(self, stroke):
        # Get similar drawings from QuickDraw dataset
        recognition = self.recognize_stroke(stroke)
        if recognition and recognition['confidence'] > 0.7:
            category = recognition['category']
            qd = QuickDrawData()
            suggestions = []
            for drawing in qd.get_drawing(category, max_drawings=3):
                suggestions.append(drawing)
            return suggestions
        return [] 