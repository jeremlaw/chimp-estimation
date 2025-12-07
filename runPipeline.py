from pipeline.loadImage import loadImage

from config import get_config
config = get_config()

json_file, image_folder, mask_folder, image_data, truth_data = loadImage(config)
