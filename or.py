import pandas as pd
from utils.utils import prepare_data, save_plot 
from utils.models import Perceptron


def main(data, model_name, plotname, eta, ephochs):
    df_or = pd.DataFrame(data)

    X,y = prepare_data(df_or)

    model_or = Perceptron(eta=eta, epochs=ephochs)
    model_or.fit(X,y)

    _ = model_or.total_loss()


    model_or.save(filename=model_name, model_dir="model")

    save_plot(df_or, model_or, filename=plotname)


if __name__ == "__main__":
    OR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=OR, model_name="or.model", plotname="or.png", eta=ETA, ephochs=EPOCHS)
