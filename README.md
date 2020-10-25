# RegpaNet

This is the code repository for the current working paper: Few-Shot Robotic Surgical Tools Segmentation based on Region Grouping and Prototype Assignment. Once the paper is accepted, details of the model will be avaiable here. Regpa means (Re)gion (g)roup (p)rototype (a)ssignment.


## Data

### EndoVis 2017

The dataset is maintained on [grand-challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/)

### ROBUST-MIS

You can obtain the dataset by following the instructions in this [synapse](https://www.synapse.org/#!Synapse:syn18779624/wiki/591266). Please noticed that our model is trained on ROBUST-MIS-2019-RELEASE.

## Model

Architecture is shown as follow. Details will be updated once the paper is accepted. Basically, our model is based on the prototype network by Jake Snell ^[J. Snell, K. Swersky, and R. Zemel, “Prototypical Networks for Few-shot Learning,” in Advances in Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds. Curran Associates, Inc., 2017, pp. 4077–4087.]. We extract support set features with the model backbone resnet-50 then obtain the prototype from the ground truth, the feature map will inject to the group as a group assignment. The assigned feature will further get the segmentation from the Query set, and it will generate a query region group. The New generated group will gain the predicted result based on the feature map extracted from the support set.

![Model](https://i.loli.net/2020/10/25/oGcgtJVdiS8OPLI.png)
