import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
from PIL import Image
from torchvision.models import vgg19_bn


class VggModel:
    def __init__(self):
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._criterion = None
        self._optimizer = None

        print(f"Using {self._device} device")

    def load_model(self, model_path=None):
        if model_path is None:
            self._model = vgg19_bn(num_classes=10).to(self._device)

        else:
            # load owned pretrained model
            self._model = vgg19_bn(num_classes=10).to(self._device)
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))

    def load_hyperparameter(
        self,
        dataset_loader,
        criterion=None,
        optimizer=None,
        epoch=60,
        lr=0.001,
        weight_decay=1e-5,
    ):
        self._criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._optimizer = (
            optimizer
            if optimizer is not None
            # else optim.SGD(self._model.parameters(), lr=lr, momentum=0.9)
            else optim.Adam(self._model.parameters(), lr=lr)
        )
        self._epoch = epoch
        self._dataset_loader = dataset_loader

    def print_model_summary(self):
        if self._model is not None:
            torchsummary.summary(self._model, (3, 32, 32))

    def calculate_average_loss_and_accuracy(self, loader):
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in loader:
                images, labels = data[0].to(self._device), data[1].to(self._device)
                outputs = self._model(images)
                loss = self._criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / len(loader), correct / total

    def train(self, print_every=2000):
        train_loader, test_loader = self._dataset_loader.get_data_loader()
        (
            self._train_avg_loss,
            self._train_accuracies,
            self._test_avg_loss,
            self._test_accuracies,
        ) = ([], [], [], [])

        for epoch in range(self._epoch):
            print(f"Epoch {epoch + 1} / {self._epoch}")
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self._device), data[1].to(self._device)

                # zero the parameter gradients
                self._optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()

                # print statistics
                if i % print_every == print_every - 1:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss:.3f}")

            # calculate accuracy and average loss and record
            (
                average_train_loss,
                train_accuracy,
            ) = self.calculate_average_loss_and_accuracy(train_loader)
            average_test_loss, test_accuracy = self.calculate_average_loss_and_accuracy(
                test_loader
            )
            print(
                f"Average train loss: {average_train_loss:.3f}, train accuracy: {train_accuracy:.3f}"
            )
            print(
                f"Average test loss: {average_test_loss:.3f}, test accuracy: {test_accuracy:.3f}"
            )

            self._train_avg_loss.append(average_train_loss)
            self._train_accuracies.append(train_accuracy)
            self._test_avg_loss.append(average_test_loss)
            self._test_accuracies.append(test_accuracy)

            # save model if test accuracy is the highest
            if test_accuracy == max(self._test_accuracies):
                print(
                    f"Saving model at epoch {epoch + 1} with test accuracy {test_accuracy:.3f}"
                )
                self.save_model()

        print("Finished Training")

    def plot_loss_and_accuracy(self):
        # plot train loss and test loss in one figure
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self._train_avg_loss, label="train")
        plt.plot(self._test_avg_loss, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.legend()

        # plot train accuracy and test accuracy in one figure
        plt.subplot(2, 1, 2)
        plt.plot(self._train_accuracies, label="train")
        plt.plot(self._test_accuracies, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()

        # save plt to file
        plt.savefig("./result.png")
        plt.show()

    def save_model(self):
        save_path = "./weights/vgg19_bn.pth"
        torch.save(self._model.state_dict(), save_path)

    def inference(self, img_path, transform):
        # load image
        img = Image.open(img_path)

        # transform
        img = transform(img)

        # add batch dimension
        img = img.unsqueeze(0)

        # inference
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(img.to(self._device))

        return outputs

    def show_inference_probability(self, outputs, classes):
        # show probability
        plt.figure(figsize=(8, 8))
        plt.bar(
            classes,
            torch.nn.functional.softmax(outputs[0], dim=0).cpu().detach().numpy(),
        )
        plt.xticks(rotation=90)
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Probability of each class")

        plt.tight_layout()
        plt.savefig("./tmp/inference.png")
        plt.show()

    def get_model(self):
        return self._model

    def get_device(self):
        return self._device
