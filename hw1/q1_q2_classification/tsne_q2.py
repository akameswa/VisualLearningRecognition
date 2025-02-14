import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
from utils import ARGS
from train_q2 import ResNet
import matplotlib.pyplot as plt
from voc_dataset import VOCDataset
from sklearn.manifold import TSNE

class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    args = ARGS(
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=0.00005,
        batch_size=32,
        step_size=10,
        gamma=0.05,
    )

    # load model
    model = ResNet(len(VOCDataset.CLASS_NAMES))
    model = model.to(args.device)

    # checkpoint loading
    ckpt = torch.load('/home/akameswa/VisualLearningRecognition/hw1/q1_q2_classification/tensorboard/q2/ckpt/checkpoint-model-epoch50.pth')
    model.load_state_dict(ckpt.state_dict())

    # create test dataset loader sampling 1000 random images
    test_dataset = VOCDataset('test', args.inp_size)
    indices = np.random.choice(len(test_dataset), 1000, replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
    )
    
    # extract features and labels
    features, labels = [], []
    model.eval()

    with torch.no_grad():
        for image, target, _ in test_loader:
            image = image.to(args.device)
            feat = model(image)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())

    features, labels = np.vstack(features), np.vstack(labels)

    # compute 2d t-SNE using sklearn projection of features 
    tsne = TSNE(n_components=2, random_state=0)
    tsne_features = tsne.fit_transform(features)

    # grab class indices
    indices = [np.where(row == 1)[0] for row in labels]

    # if multiple classes compute mean color 
    labels = [int(indices[i].mean()) if len(indices[i]) > 0 else 0 for i in range(len(indices))]

    # plot each feature color coded by GT
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                     c=labels, 
                     cmap='tab20',  
                     vmin=0,
                     vmax=len(VOCDataset.CLASS_NAMES)-1)
    
    # add legend
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=plt.cm.tab20(i/len(VOCDataset.CLASS_NAMES)),
            label=VOCDataset.CLASS_NAMES[i],
            markersize=10,
        ) for i in range(len(VOCDataset.CLASS_NAMES))
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))

    # save plot
    plt.title('t-SNE of ResNet-18 features')
    plt.tight_layout()
    plt.savefig('q1_q2_classification/tsne.png')