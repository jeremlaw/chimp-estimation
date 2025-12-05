import json

from config import get_config
config = get_config()

def loadImage(config, inputs):
    # SAMPLE DATA
    json_file = config.data['json_file']
    image_folder = config.data['images']
    mask_folder = config.data['masks']

    # load data
    with open(json_file, 'r') as file:
        image_data = json.load(file)

    # self generated pixel distances
    truth_json = '../data/red_laser_truth.json' # all data but will contain distances for samples as well

    # load truth json
    with open(truth_json, 'r') as file:
        truth_data = json.load(file)

    print(f"Running {len(image_data)} images through pipeline")





# if __name__ == "__main__":
#     import yaml
#     cfg = yaml.safe_load(open("config.yaml"))
#     out = run(cfg, {})
#     print(out)