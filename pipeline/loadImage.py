import json

def loadImage(config):
    data = config['data']
    # SAMPLE DATA
    json_file = data['json_file']
    image_folder = data['images']
    mask_folder = data['masks']

    # load data
    with open(json_file, 'r') as file:
        image_data = json.load(file)

    # load truth json (self generated pixel distances)
    with open(data['truth_json'], 'r') as file:
        truth_data = json.load(file)

    print(f"Running {len(image_data)} images through pipeline")

    return json_file, image_folder, mask_folder, image_data, truth_data




# if __name__ == "__main__":
#     import yaml
#     cfg = yaml.safe_load(open("config.yaml"))
#     out = run(cfg, {})
#     print(out)