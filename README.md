# CASA0018-Apple-Classification

## Introduction
Being able to correctly identify apple type, fruit type, and other shopping items is important for improving the self-service experience in supermarkets. The future of supermarket self-checkouts will involve ai kiosks using computer vision to correctly label food items without the need for barcode scanning or manually inputting your order and quantity into the self-checkout kiosk. This AI approach helps to reduce the queue times in the self-checkout zone. 

Supermarkets in the United Kingdom have approximately £3.2 of goods stolen annually at the self-checkout. The use of computer-vision techniques in self-checkout kiosks could introduce new preventative measures against theft. Missed scans comprise a large portion of how goods go unpaid. 20-30% of missed scans are intentional. Customers also attempt to dupe the system by weighing an incorrect item for a lower price. The use of computer vision helps to reduce missed scans and catch customers aiming to trick the system (Gitnux, 2023).

## Literature Review
Pre-trained CNNs are typically trained on massive datasets containing millions of labelled images, covering a wide range of object types. The early layers within these pre-trained models detect low-level features such as textures and edges. Deeper levels recognise more complex features. The final layers of the pre-trained model are fine-tuned and trained on a new dataset for more accurate predictions on the new target classes. As a result of this setup, there are various benefits of using transfer learning over training a CNN from scratch. The knowledge obtained in the pre-trained CNN is maintained and improves the predictive accuracy of the new target class. The model requires less training data than building a CNN from scratch due to generalizability from the pre-training and reduces the probability of overfitting. This results in a less computationally expensive model to train for faster iterations to improve predictive accuracy. 

Research done by Yonis Gulzar in his paper “Fruit Image Classification Model Based on MobileNetV2 with Deep Transfer Learning Technique” found that the best CNN architecture for image classification on their fruit dataset was MobileNetV2. This bar chart shows the accuracy of different models achieved while training on the fruit dataset containing forty different types of fruits. As shown in Figure 1, MobileNetV2 achieves the highest accuracy of 89%. The remaining pre-trained CNNs were 5-11% lower in accuracy than MobileNetV2.
<div align=center>   
![Pre-trained CNN comparison](https://user-images.githubusercontent.com/73647889/232113670-353c7cc8-0e5e-43a8-8cfe-37e1697f15e4.PNG)

## MobileNetV2 Neural Architecture
MobileNetV2 was designed with resource constrained environments such as embedded devices and mobile phones. The neural architecture is as follows:

Inverted Residuals: This design, a variation of residual connections in ResNet, reverses the order of bottleneck and wider layers. Inverted residuals consist of an expansion layer, which increases channels, a depthwise convolutional layer for capturing spatial information, and a projection layer, which reduces channels while preserving spatial resolution. This results in reduced computational complexity and enhanced information flow.

Linear Bottlenecks: By omitting ReLU activation from the bottleneck layers' end, MobileNetV2 preserves crucial information that might otherwise be lost, ensuring predictive accuracy. This results in a compact and efficient model that can still learn complex patterns, striking an ideal balance between efficiency and performance.

Depthwise Separable Convolutions: MobileNetV2 leverages "depthwise separable convolutions" to minimize computational complexity compared to standard convolutions. By breaking down the convolution operation into depthwise and pointwise (1x1) convolutions, the network simplifies the process and cuts down on required computations, creating a more efficient and lightweight architecture.

## Data Collection
![Apple Photos](https://user-images.githubusercontent.com/73647889/232113757-94f8452e-5c73-4d7e-8f48-e0de8d2b6a75.PNG)

