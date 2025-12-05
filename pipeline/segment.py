



# pipeline/segment.py
def run(config, inputs):
    # do segmentation using inputs or config
    segments = ...  # compute
    return {"segments": segments}

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    out = run(cfg, {})
    print(out)