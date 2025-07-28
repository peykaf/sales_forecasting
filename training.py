from predictive.catboost_regressor_model import CatboostRegressorModel


def main():
    model = CatboostRegressorModel()
    model.train()

if __name__ == "__main__":
    main()


