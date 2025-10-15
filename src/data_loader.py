import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class DataLoader:
    def __init__(self, img_height, img_width, batch_size, data_root_dir):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.data_root_dir = data_root_dir # This is 'xray_classifier/data'

        self.train_dir = os.path.join(self.data_root_dir, 'raw', 'train')
        self.val_dir = os.path.join(self.data_root_dir, 'raw', 'val')
        self.test_dir = os.path.join(self.data_root_dir, 'raw', 'test')

        self.train_datagen = None
        self.val_datagen = None
        self.test_datagen = None
        self.class_names = [] # To store class names from the generator

    def create_generators(self, augmentation=True):
        print(f"Loading data from: {self.data_root_dir}")
        print(f"Train directory: {self.train_dir}")
        print(f"Validation directory: {self.val_dir}")
        print(f"Test directory: {self.test_dir}")


        if augmentation:
            self.train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.train_datagen = ImageDataGenerator(rescale=1./255)

        self.val_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        # Check if directories exist
        if not os.path.exists(self.train_dir) or not os.listdir(self.train_dir):
            raise FileNotFoundError(f"Training data directory not found or empty: {self.train_dir}")
        if not os.path.exists(self.val_dir) or not os.listdir(self.val_dir):
            raise FileNotFoundError(f"Validation data directory not found or empty: {self.val_dir}")
        if not os.path.exists(self.test_dir) or not os.listdir(self.test_dir):
            raise FileNotFoundError(f"Test data directory not found or empty: {self.test_dir}")


        train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary' # Change to 'categorical' for multi-class classification
        )

        val_generator = self.val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary' # Change to 'categorical' for multi-class classification
        )

        test_generator = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary', # Change to 'categorical' for multi-class classification
            shuffle=False # Keep data in order for evaluation metrics
        )

        self.class_names = list(train_generator.class_indices.keys())
        print(f"Class names found: {self.class_names}")

        return train_generator, val_generator, test_generator

if __name__ == "__main__":
    # This block is for testing the DataLoader independently.
    # It creates dummy data to ensure the data loader works.
    # In a real project, you would remove or comment out the dummy data creation.

    print("Running DataLoader test block...")
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    BATCH_SIZE = 32
    # Adjust DATA_ROOT to point to 'xray_classifier/data' from the current script location
    DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # Create dummy directories and files for testing (remove in real usage)
    print(f"Creating dummy data in: {DATA_ROOT}/raw")
    dummy_train_normal_dir = os.path.join(DATA_ROOT, 'raw', 'train', 'normal')
    dummy_train_condition_dir = os.path.join(DATA_ROOT, 'raw', 'train', 'condition')
    dummy_val_normal_dir = os.path.join(DATA_ROOT, 'raw', 'val', 'normal')
    dummy_val_condition_dir = os.path.join(DATA_ROOT, 'raw', 'val', 'condition')
    dummy_test_normal_dir = os.path.join(DATA_ROOT, 'raw', 'test', 'normal')
    dummy_test_condition_dir = os.path.join(DATA_ROOT, 'raw', 'test', 'condition')

    os.makedirs(dummy_train_normal_dir, exist_ok=True)
    os.makedirs(dummy_train_condition_dir, exist_ok=True)
    os.makedirs(dummy_val_normal_dir, exist_ok=True)
    os.makedirs(dummy_val_condition_dir, exist_ok=True)
    os.makedirs(dummy_test_normal_dir, exist_ok=True)
    os.makedirs(dummy_test_condition_dir, exist_ok=True)

    from PIL import Image
    for i in range(5): # Create a few dummy images
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (i*50 % 255, 0, 0)).save(os.path.join(dummy_train_normal_dir, f'img_{i}.png'))
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (0, i*50 % 255, 0)).save(os.path.join(dummy_train_condition_dir, f'img_{i}.png'))
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (0, 0, i*50 % 255)).save(os.path.join(dummy_val_normal_dir, f'img_{i}.png'))
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (i*50 % 255, i*50 % 255, 0)).save(os.path.join(dummy_val_condition_dir, f'img_{i}.png'))
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (0, i*50 % 255, i*50 % 255)).save(os.path.join(dummy_test_normal_dir, f'img_{i}.png'))
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = (i*50 % 255, 0, i*50 % 255)).save(os.path.join(dummy_test_condition_dir, f'img_{i}.png'))
    print("Dummy data created for testing DataLoader.")

    try:
        data_loader = DataLoader(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, DATA_ROOT)
        train_gen, val_gen, test_gen = data_loader.create_generators()

        print(f"\nNumber of training batches: {len(train_gen)}")
        print(f"Number of validation batches: {len(val_gen)}")
        print(f"Number of test batches: {len(test_gen)}")
        print(f"First training batch shape: {next(train_gen)[0].shape}") # (batch_size, img_height, img_width, 3)
        print(f"Class names reported by DataLoader: {data_loader.class_names}")

    except FileNotFoundError as e:
        print(f"Caught expected error (if actual data is missing): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up dummy data
        import shutil
        dummy_raw_dir = os.path.join(DATA_ROOT, 'raw')
        if os.path.exists(dummy_raw_dir):
            shutil.rmtree(dummy_raw_dir)
            print("Dummy data cleaned up.")