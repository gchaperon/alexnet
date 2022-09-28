import alexnet.data as data
import alexnet.models as models


def main():

    model = models.AlexNet(nclasses=data.TinyImageNet.nclasses)

    datamodule = data.LitTinyImageNet(datadir="data")
    datamodule.setup("validate")



if __name__ == "__main__":
    main()
