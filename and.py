import pandas as pd
from utils.utils import prepare_data, save_plot 
from utils.models import Perceptron
import logging
import os

gate = "AND gate"
log_dir ="logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
    )

def main(data, model_name, plotname, eta, ephochs):
    df = pd.DataFrame(data)
    logging.info(f"This is the Raw Dataset:{df}")
    X,y = prepare_data(df)

    model_and = Perceptron(eta=eta, epochs=ephochs)
    model_and.fit(X,y)

    _ = model_and.total_loss()


    model_and.save(filename=model_name, model_dir="model")

    save_plot(df, model_and, filename=plotname)


if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y":  [0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info(f">>>>>>>>>>> starting training for {gate} >>>>>>>>>>>>>>>>>>")
        main(data=AND, model_name="and.model", plotname="and.png", eta=ETA, ephochs=EPOCHS)
        logging.info(f"<<<<<<<<<<< Complate training for {gate} <<<<<<<<<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
