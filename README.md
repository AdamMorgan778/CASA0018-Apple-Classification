# CASA0018-Apple-Classification

## Introduction
Being able to correctly identify apple type, fruit type, and other shopping items is important for improving the self-service experience in supermarkets. The future of supermarket self-checkouts will involve ai kiosks using computer vision to correctly label food items without the need for barcode scanning or manually inputting your order and quantity into the self-checkout kiosk. This AI approach helps to reduce the queue times in the self-checkout zone. 

Supermarkets in the United Kingdom have approximately £3.2 of goods stolen annually at the self-checkout. The use of computer-vision techniques in self-checkout kiosks could introduce new preventative measures against theft. Missed scans comprise a large portion of how goods go unpaid. 20-30% of missed scans are intentional. Customers also attempt to dupe the system by weighing an incorrect item for a lower price. The use of computer vision helps to reduce missed scans and catch customers aiming to trick the system (Gitnux, 2023).

## Literature Review
Pre-trained CNNs are typically trained on massive datasets containing millions of labelled images, covering a wide range of object types. The early layers within these pre-trained models detect low-level features such as textures and edges. Deeper levels recognise more complex features. The final layers of the pre-trained model are fine-tuned and trained on a new dataset for more accurate predictions on the new target classes. As a result of this setup, there are various benefits of using transfer learning over training a CNN from scratch. The knowledge obtained in the pre-trained CNN is maintained and improves the predictive accuracy of the new target class. The model requires less training data than building a CNN from scratch due to generalizability from the pre-training and reduces the probability of overfitting. This results in a less computationally expensive model to train for faster iterations to improve predictive accuracy. 

Research done by Yonis Gulzar in his paper “Fruit Image Classification Model Based on MobileNetV2 with Deep Transfer Learning Technique” found that the best CNN architecture for image classification on their fruit dataset was MobileNetV2. This bar chart shows the accuracy of different models achieved while training on the fruit dataset containing forty different types of fruits. As shown in Figure 1, MobileNetV2 achieves the highest accuracy of 89%. The remaining pre-trained CNNs were 5-11% lower in accuracy than MobileNetV2.

![Pre-trained CNN comparison](https://user-images.githubusercontent.com/73647889/232118596-8ca3d43c-acaf-49ff-bddb-e8c24997dad1.PNG)

## MobileNetV2 Neural Architecture
MobileNetV2 was designed with resource constrained environments such as embedded devices and mobile phones. The neural architecture is as follows:

Inverted Residuals: This design, a variation of residual connections in ResNet, reverses the order of bottleneck and wider layers. Inverted residuals consist of an expansion layer, which increases channels, a depthwise convolutional layer for capturing spatial information, and a projection layer, which reduces channels while preserving spatial resolution. This results in reduced computational complexity and enhanced information flow.

Linear Bottlenecks: By omitting ReLU activation from the bottleneck layers' end, MobileNetV2 preserves crucial information that might otherwise be lost, ensuring predictive accuracy. This results in a compact and efficient model that can still learn complex patterns, striking an ideal balance between efficiency and performance.

Depthwise Separable Convolutions: MobileNetV2 leverages "depthwise separable convolutions" to minimize computational complexity compared to standard convolutions. By breaking down the convolution operation into depthwise and pointwise (1x1) convolutions, the network simplifies the process and cuts down on required computations, creating a more efficient and lightweight architecture.

<img width="251" alt="MobileNetV2" src="https://user-images.githubusercontent.com/73647889/232118629-3c9b290b-a688-4ebb-bcce-d3648d6fc532.png">

## Data Collection
The data was collected in batches of 40 photos containing a 50% split between lady pink and granny smith apples. I increased the training data incrementally until the training set was 200 photos. The test set contained a total of 35 photos. This allowed me to see how results improve as the training data set increases.

The photos were taken using the OV7675 camera – The photos image width was 160 and image height was 120 before being resized to 96x96 to be compatible with transfer learning, the smaller image size also makes the process of image classification less resource hungry. The data were then pre-processed and normalized. Colour plays a large role in the classification of apples, so the colour depth was set to RGB.

Photos were taken with different backgrounds, on different surfaces, in light conditions, various camera distances from the apple, and a variety of angles. The figure below showcases a few of the pictures taken with the OV7675 in the training dataset.

![Apple Photos](https://user-images.githubusercontent.com/73647889/232118661-cec8043e-4380-4b00-b42e-8d301534d8d4.PNG)


## Testing
A total of 47 experiments were done to find the best algorithm for classifying pink lady and granny smith apples. Here are some discoveries I made during experimenting and tweaking the hyperparameters and training set size of the model:
  
Due to time constraints it was difficult to take 100s of photos, to overcome the small amount of training data I used data augmentation. This method creates random artificial images from the source data using various methods such as random rotations, shifts, and shears. I found the model only benefited from data augmentation when the number of epochs was increased otherwise data augmentation negatively impacted on predicting test data in my experiments. 

The models listed below share identical hyperparameters and were compared in terms of on-device performance on the Arduino Nano 33 BLE Sense (Cortex-M4F 64MHz). MobileNetV1 96x96 0.1 was ultimately selected due to its high accuracy and fast inferencing time. MobileNetV2, by contrast, exhibited a slower inferencing time. This discrepancy could be attributed to the more complex neural network architecture of V2 compared to V1. Another significant factor influencing inferencing time was the width multiplier; lower values resulted in faster inferencing times. Inferencing time is crucial in self-checkout applications, as it helps reduce queue times and provides a more responsive, improved user experience. Interestingly, MobileNetV1 outperformed MobileNetV2 in both accuracy and inferencing time while using the same hyperparameters and consuming fewer resources, as illustrated in the figure below.

By using live classification, I could identify biases in the training dataset, I discovered that red apples would be classified as green if it appears in the top lefthand corner of the image. To counter this I took more images of red apples where the apple appeared in the top lefthand corner. I also discovered that green backgrounds would influence the model to predict red apples as green, so I collected more training data to counter this.

## Reflection
A more systematic approach to testing could have been beneficial, as various transfer learning models were only explored in the 38th experiment and beyond due to preconceived notions about MobileNetV2's effectiveness. Testing different models earlier might have helped identify the best option sooner, allowing more time for fine-tuning. By tracking inferencing time, peak RAM usage, and flash usage from the beginning, it would be possible to better understand how different models and hyperparameters impact on-device performance and further reduce inference time.

More time should have been spent testing the model outside of the edge impulse transfer learning and model testing section. To increase the model's robustness, additional edge cases should have been considered to ensure accurate labelling. Scenarios not taken into account include instances where no apple is present, situations where the camera might misclassify an orange as an apple, or instances where the model might incorrectly classify a golden apple as a pink lady or granny smith.
  
## Future
•	Creating a significantly larger test set to fine tune the model further. Currently the model predicts all 40 of the test-set correctly so any improvements to the model are not recognisable.

•	Try using MobileNetV3 for potentially greater accuracy and reduced latency. Reduced latency is the largest concern as the accuracy is already very high. The latency is currently 1.9 seconds which is quite slow for detecting apple type in a self-checkout setting.

•	Create training dataset that more accurately simulate a self-checkout setting

•	Increase the number of items able to be classified.

•	Identify when there is no object present

  

