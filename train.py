from question_5.DatasetLoader import DatasetLoader
from question_5.VggModel import VggModel

if __name__ == "__main__":
    vgg_model = VggModel()
    dataset_loader = DatasetLoader(batch_size=128, data_augmentation_flag=True)

    # load pretrained model
    vgg_model.load_model()

    """
    load dataset
    """
    # dataset_loader.load_cifar10_dataset(clip_dataset_count=1000)
    dataset_loader.load_cifar10_dataset()

    """
    train model
    """
    vgg_model.load_hyperparameter(dataset_loader, epoch=100, lr=2e-4)

    vgg_model.train()

    vgg_model.plot_loss_and_accuracy()
