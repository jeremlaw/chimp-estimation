# chimp-estimation

Fuzzy Wuzzy pipeline recreation
Team Inchworm


Architecture outline:

pipeline_recreation
    data (holds different datasets for us to use)
        smallTest (repeatable example)
            images
                .jpgs
            masks
                .jpgs
            measured.csv
            data.json
            truth.json
    pipeline (steps for pipeline)
        loadImage.py
        segment.py
        detectLaser.py
        estimatePose.py
        estimateSize.py
    runPipeline.py (file to run all steps)
    config.yaml (file to hold preferences for the pipeline)
    setup.py (file to ensure proper models are present and configuration completed)
