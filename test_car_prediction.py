import unittest
import numpy as np
import pickle  # or joblib if you've used that
import os
from Ridge import Ridge
from Ridge import RidgePenalty
import sys
import mlflow
from dotenv import load_dotenv

class TestCarPredictionModel(unittest.TestCase):
    
    def setUp(self):
        # Load the pre-trained model
                
        sys.modules['__main__'].Ridge = Ridge
        sys.modules['__main__'].RidgePenalty = RidgePenalty

    def setUp(self):
        # The MLflow model URI
        load_dotenv()
        # Access environment variables
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        model_uri = f'http://{username}:{password}@mlflow.ml.brain.cs.ait.ac.th/#/experiments/217142467269316447/runs/05ea8a55fef44f4d8b14cd6e78c4fc9d/artifacts/model/model.pkl'

        # Load the model from MLflow
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            self.fail(f"Failed to load the model from MLflow: {str(e)}")

        # Define a valid input for testing
        self.valid_input = [60000.0, 1, 19.38, 82.85, 1248.0]  # Example valid input

    def test_model_input_format(self):
        """Test that the model accepts the expected input format."""
        try:
            output = self.model.predict([self.valid_input])
            self.assertTrue(True)  # If no exception, the input is valid
        except Exception as e:
            self.fail(f"Model failed to accept valid input. Error: {str(e)}")

    def test_model_output_shape(self):
        """Test that the model returns an output of expected shape."""
        output = self.model.predict([self.valid_input])
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1,))

if __name__ == "__main__":
    unittest.main()
