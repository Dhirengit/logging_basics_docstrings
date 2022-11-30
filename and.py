import pandas as pd
from utils.utils import prepare_data, save_plot 
from utils.models import Perceptron


def main(data, model_name, plotname, eta, ephochs):
    df_and = pd.DataFrame(data)

    X,y = prepare_data(df_and)

    model_and = Perceptron(eta=eta, epochs=ephochs)
    model_and.fit(X,y)

    _ = model_and.total_loss()


    model_and.save(filename=model_name, model_dir="model")

    save_plot(df_and, model_and, filename=plotname)


if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y":  [0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=AND, model_name="and.model", plotname="and.png", eta=ETA, ephochs=EPOCHS)
