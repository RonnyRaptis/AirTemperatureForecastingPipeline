# main.py
import os
import logging
import data_loader
import train
import evaluate
import predict
import config

def setup_directories() -> None:
    """Ensure required directories exist."""
    for directory in [config.DATA_DIR, config.OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info("Created directory: %s", directory)
        else:
            logging.info("Directory exists: %s", directory)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    setup_directories()
    
    try:
        # Step 1: Download and process the data
        data_loader.download_data()
        data_loader.process_data()
    except Exception as e:
        logging.error("Data loading/processing failed: %s", e)
        return
    
    try:
        # Step 2: Train the model
        train.run_training()
    except Exception as e:
        logging.error("Training failed: %s", e)
        return

    try:
        # Step 3: Evaluate the model
        evaluate.run_evaluation()
    except Exception as e:
        logging.error("Evaluation failed: %s", e)
    
    try:
        # Step 4: Make predictions
        predict.run_prediction()
    except Exception as e:
        logging.error("Prediction failed: %s", e)

if __name__ == '__main__':
    main()
