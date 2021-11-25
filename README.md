# Colorizer

A project to bulid and train model to color grayscale images. 

It is a customizable and completetly abstract project capable of running a base model (resnet18). The base model was trained for 100 epochs and the `state_dict` stored in `colorizer/serialized/colornet18_ckpt_pth`.

The componients are implemented using factory design pattern and config.py to provide a layer of abstractness and easy extensibility.

## How to run
**Note: Serialized state dict present in `colorizer/serialized/colornet18_ckpt.pth` is a serialized version of the base model trained for 100 epochs. One can directly run inference using the default configs (just use the cli command in inference mode, the result will be saved in `colorized/output/colored.jpg`).**

**Step 1.** Set up a `conda` environment using `requirements.txt`

- The code can be run directly using CLI mode as -

  First, make the required changes in hyperparameters in `config.py` and run the following.
  **Make sure you are in `image-colorization` directory in terminal**
  - For training:
    ```sh
    python3 colorize.py --mode=train
    ```
  - For inference: If `state_dict_path` is not provided in `argv` the path provided in `config.py` will be used.
    - If you want to use the default `state_dict`:  
    ```sh
    python3 colorize.py --mode=inference --image_path=`PATH/TO/IMAGE/FOR/INFERENCE`
    ```
    - If you have you own trained `state_dict` for base_model:
    ```sh
    python3 colorize.py --mode=inference --image_path=`PATH/TO/IMAGE/FOR/INFERENCE` --state_dict_path=`PATH/TO/IMAGE/FOR/INFERENCE`
    ```

- To run the model using code see `example.py`.


## Models
- **ResNet18:** The base model using `ResNet18` was completed.
- **ConstantNet:** The idea is to create a "not too deep" network which passes on from input to output without changing the shape of image i.e, keeping the height and width constant. Just varying the information spatially depthwise. As I exhausted my Kaggle quota, I was able to train this network for only 30 epochs, the result showed good structural similarity, but the colors were not that great.
  - Ideology:
    - We know CNNs are good feature extractors. Where the lower layers (nearer to the input) extract pixel level features such as borders, colors etc, and the higher layers (further from input) extract more structural content of the image (higher features).
    - In this use case we do not want to in any way modify the structural information, we rather want to modify / add information in the depth / color channel. So, it makes sense to only work with the starting layers and information content.
    - One might argue that keeping the dimensions unchanges can be computationally expensive, but we compensate that by building a "not too deep" network, as we are not concerned about varying the structure level features of image.
  - This is just a theoretical model that I build on the above understandings.
- Further possible models:
  - **ResNet18 + Cross connections:** Using the pretrained resnet18 as base network and adding skip connections (cross-connections) from the encoder block to the decoder block.
  - **ResNet50 / DenseNet121:** Using a deeper pre-trained base network.
  - Hybrid of deeper pretrained network and cross connections
  - **VGG16 + Cross connections:** Well, we cannot leave the VGG behind, I think using VGG encoder along with cross connections would give good result but the downside begin computationally very expensive!


## Loss
- **Mean Squared Error Loss:** Started by implementing MSE loss as it is a regression problem, it seemed feasible to start with it.
- **Huble Loss:** It is a similar loss to MSE, but is more robust to outliers.
  - The "good" thing about this loss is it is a combination of L1 and L2 loss depending on the margin.
  - Why I think it is a "good" metric?
    - When the absolute difference between predicted and colored image is less than a certain margin, we use L2 loss.
    - But when the absolute difference is larger, then that must be due to the "structural dissimilarity" between the images. We know that L1 loss proves to be good (if not the best) when images differ structurally.
- **Perpectual Loss:** Deriving inspiration from Neural Style Transfer, built the perpectual loss function.
  - It seemed reasonable as we wanted the predicted image to have same style as well as same content as the colored target.
  - This is a computationally expensive function, as we need an auxiliary trained network (here, VGG16) to extract intermediate features for images. I was unable to train using this loss as I do not own any GPUs, moreover GPU on Kaggle kept running OOM.
- **Contrastive Loss:** Contrastive loss has not only found wide usage (from training siamese network to dehazing images) but also is providing good results.
  - It seemed reasonable to use contrastive loss, because we wanted our predicted image to very similar to colored image but far away from grayscale image.
  - There can be several ways of implementing it:
    - Getting a feature vector from a pre-trained network and calculating contrastive loss
    - Getting intermediate features from a pre-trained network, finding distances between them. Using this distance to calculate contrastive loss and doing a weighted sum of loss from each intermediate feature.
    - This is also a computationally expensive function, as we need an auxiliary trained network (here, VGG16) to extract intermediate features for images. I was unable to train using this loss as I do not own any GPUs, moreover GPU on Kaggle kept running OOM.

## Metrics
- **Mean Squared Error:** It is a good starting metric to see how the predicted image is compared to the colored target. But it does not provide any information about how model performs in each channel.
  - It is an inverse metric, i.e, the smaller the value the better the performance of model.
- **Channelwise L2:** Calculating L2 loss channel-wise and adding them. This provides a bit more information about model's performance in.
  - This is also an inverse metric. The smaller the better.
- **PSNR:** As we know PSNR is generally used to quantify reconstruction quality of image, hence this help provide more information on our models performance.
  - It is a normal metric, i.e, the larger the better.
  - Since, it provides a measure of reconstruction quality, we can also access the structural quality of the predicted image.



