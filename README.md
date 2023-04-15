# CASA0018-Apple-Classification
## Objective
The objective of my project is to employ computer vision for accurate classification of apples as either Granny Smith or Pink Lady using an Arduino Nano 33 BLE Sense Lite. This model is intended for use in supermarket self-checkout systems. The ultimate goal is to achieve high prediction accuracy while maintaining low inference times.

## Introduction
Being able to correctly identify apple type, fruit type, and other shopping items is important for improving the self-service experience in supermarkets. The future of supermarket self-checkouts will involve ai kiosks using computer vision to correctly label food items without the need for barcode scanning or manually inputting your order and quantity into the self-checkout kiosk. This AI approach helps to reduce the queue times in the self-checkout zone. 

Supermarkets in the United Kingdom have approximately £3.2 of goods stolen annually at the self-checkout. The use of computer-vision techniques in self-checkout kiosks could introduce new preventative measures against theft. Missed scans comprise a large portion of how goods go unpaid. 20-30% of missed scans are intentional. Customers also attempt to dupe the system by weighing an incorrect item for a lower price. The use of computer vision helps to reduce missed scans and catch customers aiming to trick the system (Gitnux, 2023).



## Literature Review
Pre-trained CNNs are typically trained on massive datasets containing millions of labelled images, covering a wide range of object types. The early layers within these pre-trained models detect low-level features such as textures and edges. Deeper levels recognise more complex features. The final layers of the pre-trained model are fine-tuned and trained on a new dataset for more accurate predictions on the new target classes. As a result of this setup, there are various benefits of using transfer learning over training a CNN from scratch. The knowledge obtained in the pre-trained CNN is maintained and improves the predictive accuracy of the new target class. The model requires less training data than building a CNN from scratch due to generalizability from the pre-training and reduces the probability of overfitting. This results in a less computationally expensive model to train for faster iterations to improve predictive accuracy. 

Research done by Yonis Gulzar in his paper “Fruit Image Classification Model Based on MobileNetV2 with Deep Transfer Learning Technique” found that the best CNN architecture for image classification on their fruit dataset was MobileNetV2. This bar chart shows the accuracy of different models achieved while training on the fruit dataset containing forty different types of fruits. As shown in Figure 1, MobileNetV2 achieves the highest accuracy of 89%. The remaining pre-trained CNNs were 5-11% lower in accuracy than MobileNetV2.

![image](https://user-images.githubusercontent.com/73647889/232231429-0406aee2-7172-4090-811e-f776cc5ce0fc.png)

## MobileNetV2 Neural Architecture
MobileNetV2 was designed with resource constrained environments such as embedded devices and mobile phones. The neural architecture is as follows:

1. Inverted Residuals: This design, a variation of residual connections in ResNet, reverses the order of bottleneck and wider layers. Inverted residuals consist of an expansion layer, which increases channels, a depthwise convolutional layer for capturing spatial information, and a projection layer, which reduces channels while preserving spatial resolution. This results in reduced computational complexity and enhanced information flow.

2. Linear Bottlenecks: By omitting ReLU activation from the bottleneck layers' end, MobileNetV2 preserves crucial information that might otherwise be lost, ensuring predictive accuracy. This results in a compact and efficient model that can still learn complex patterns, striking an good balance between efficiency and performance.

3. Depthwise Separable Convolutions: MobileNetV2 leverages "depthwise separable convolutions" to minimize computational complexity compared to standard convolutions. By breaking down the convolution operation into depthwise and pointwise (1x1) convolutions, the network simplifies the process and cuts down on required computations, creating a more efficient and lightweight architecture.

<img width="251" alt="MobileNetV2" src="https://user-images.githubusercontent.com/73647889/232118629-3c9b290b-a688-4ebb-bcce-d3648d6fc532.png">

## Data Collection
The data was collected in batches of 40 photos containing a 50% split between lady pink and granny smith apples. I increased the training data incrementally until the training set was 200 photos. The test set contained a total of 35 photos. This allowed me to see how results improve as the training data set increases.

The photos were taken using the OV7675 camera – The photos image width was 160 and image height was 120 before being resized to 96x96 to be compatible with transfer learning, the smaller image size also makes the process of image classification less resource hungry. The data were then pre-processed and normalized. Colour plays a large role in the classification of apples, so the colour depth was set to RGB.

Photos were taken with different backgrounds, on different surfaces, in light conditions, various camera distances from the apple, and a variety of angles. The figure below showcases a few of the pictures taken with the OV7675 in the training dataset.

![Apple Photos](https://user-images.githubusercontent.com/73647889/232118661-cec8043e-4380-4b00-b42e-8d301534d8d4.PNG)


## Testing
A total of 47 experiments were conducted to identify the optimal algorithm for classifying Pink Lady and Granny Smith apples. Throughout the experimentation process, various hyperparameters, training set sizes, and transfer learning models were adjusted and compared. Key insights gained during the experimentation are detailed below, along with a table showcasing the various models and hyperparameters tested:


| Category          | Setting                              |
| ----------------- | ------------------------------------ |
| Training data     | 40, 80, 120, 160, 200                |
| Model             | MobileNetV1, MobileNetV2             |
| Resolution        | 96x96                                |
| Width multiplier  | 0.2, 0.25, 0.35                      |
| Neurons           | 0, 4, 8, 12, 16, 20, 24, 48, 62, 128 |
| Epochs            | 20, 40, 50, 60, 70, 100, 120, 160    |
| Dropout           | 0.1, 0.3                             |
| data augmentation | off, on                              |

  
  
Due to time constraints it was difficult to take 100s of photos, to overcome the small amount of training data I used data augmentation. This method creates random artificial images from the source data using various methods such as random rotations, shifts, and shears. I found the model only benefited from data augmentation when the number of epochs was increased otherwise data augmentation negatively impacted on predicting test data in my experiments. The increase in training data led to improved accuracy, but the gains diminished after reaching 120 training photos. The figure below illustrates the enhancement in model performance when using data augmentation compared to without it.


![image](https://user-images.githubusercontent.com/73647889/232231518-2bbe0d46-5ffa-42e3-b373-2562ee8f209a.png)


The models listed below share identical hyperparameters and were compared in terms of on-device performance on the Arduino Nano 33 BLE Sense (Cortex-M4F 64MHz). MobileNetV1 96x96 0.1 was ultimately selected due to its high accuracy and fast inferencing time. MobileNetV2, by contrast, exhibited a slower inferencing time. This discrepancy could be attributed to the more complex neural network architecture of V2 compared to V1. Another significant factor influencing inferencing time was the width multiplier; lower values resulted in faster inferencing times. Inferencing time is crucial in self-checkout applications, as it helps reduce queue times and provides a more responsive, improved user experience. Interestingly, MobileNetV1 outperformed MobileNetV2 in both accuracy and inferencing time while using the same hyperparameters and consuming fewer resources, as illustrated in the table below.


| Model                  | Inferencing time (ms) | Peak ram usage (K) | Flash usage (K) | Accuracy (%) | Loss | Test Accuracy (%) |
| ---------------------- | --------------------- | ------------------ | --------------- | ------------ | ---- | ----------------- |
| MobileNetV2 96x96 0.35 | 1,978                 | 333.7              | 581.8           | 100          | 0.04 | 92.86             |
| MobileNetV2 96x96 0.1  | 961                   | 279.9              | 219             | 100          | 0.01 | 92.86             |
| MobileNetV2 96x96 0.05 | 825                   | 270                | 169             | 100          | 0    | 92.86             |
| MobileNetV1 96x96 0.2  | 812                   | 100.1              | 224.4           | 100          | 0.04 | 96.43             |
| MobileNetV1 96x96 0.1  | 207                   | 59.9               | 107             | 100          | 0.04 | 100               |


By using live classification, I could identify biases in the training dataset, I discovered that red apples would be classified as green if it appears in the top lefthand corner of the image. To counter this I took more images of red apples where the apple appeared in the top lefthand corner. I also discovered that green backgrounds would influence the model to predict red apples as green, so I collected more training data to counter this.

### All Experiments

| Test Number | Training data (n) | NN type     | Resolution | Width Multiplier | Neurons | drop out | Epochs | Learning rate | validation (%) | data augmentation | Accuracy | Loss | Testing Accuracy |
| ----------- | ----------------- | ----------- | ---------- | ---------------- | ------- | -------- | ------ | ------------- | -------------- | ----------------- | -------- | ---- | ---------------- |
| 1           | 40                | MobileNetV2 | 96x96      | 0.35             | 16      | 0.1      | 20     | 0.0005        | 20             | off               | 75.00%   | 0.56 | 72%              |
| 2           | 40                | MobileNetV2 | 96x96      | 0.35             | 16      | 0.1      | 20     | 0.0005        | 20             | on                | 75.00%   | 0.55 | 70%              |
| 3           | 40                | MobileNetV2 | 96x96      | 0.35             | 16      | 0.1      | 20     | 0.0005        | 20             | on                | 75.00%   | 0.66 | 68%              |
| 4           | 40                | MobileNetV2 | 96x96      | 0.35             | 16      | 0.1      | 20     | 0.0005        | 20             | off               | 75.00%   | 0.56 | 72%              |
| 5           | 40                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 20             | off               | 87.50%   | 0.54 | 80%              |
| 6           | 40                | MobileNetV2 | 96x96      | 0.35             | 4       | 0.1      | 20     | 0.0005        | 20             | off               | 87.50%   | 0.43 | 66%              |
| 7           | 40                | MobileNetV2 | 96x96      | 0.35             | 8       | 0.1      | 20     | 0.0005        | 20             | on                | 62.50%   | 0.74 | 44%              |
| 8           | 40                | MobileNetV2 | 96x96      | 0.35             | 12      | 0.1      | 20     | 0.0005        | 20             | on                | 87.50%   | 0.49 | 58%              |
| 9           | 40                | MobileNetV2 | 96x96      | 0.35             | 20      | 0.1      | 20     | 0.0005        | 20             | off               | 80.00%   | 0.29 | 78%              |
| 10          | 40                | MobileNetV2 | 96x96      | 0.35             | 20      | 0.1      | 20     | 0.0005        | 20             | off               | 80.00%   | 0.27 | 80%              |
| 11          | 40                | MobileNetV2 | 96x96      | 0.35             | 24      | 0.1      | 20     | 0.0005        | 20             | off               | 40.00%   | 1.13 | 68%              |
| 12          | 40                | MobileNetV2 | 96x96      | 0.35             | 48      | 0.1      | 20     | 0.0005        | 20             | off               | 60.00%   | 0.91 | 72%              |
| 13          | 40                | MobileNetV2 | 96x96      | 0.35             | 62      | 0.1      | 20     | 0.0005        | 20             | off               | 60.00%   | 0.93 | 72%              |
| 14          | 40                | MobileNetV2 | 96x96      | 0.35             | 128     | 0.1      | 20     | 0.0005        | 20             | off               | 60.00%   | 0.64 | 74%              |
| 15          | 40                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 50     | 0.0005        | 20             | off               | 87.50%   | 0.54 | 80%              |
| 16          | 40                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 90     | 0.0005        | 20             | off               | 87.50%   | 0.54 | 80%              |
| 17          | 40                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.3      | 20     | 0.0005        | 20             | off               | 87.50%   | 0.57 | 80%              |
| 18          | 40                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 10             | off               | 75.00%   | 0.24 | 80%              |
| 19          | 80                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 20             | off               | 100.00%  | 0.06 | 92%              |
| 20          | 80                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | off               | 87.50%   | 0.11 | 92%              |
| 21          | 80                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 15             | off               | 91.70%   | 0.19 | 94%              |
| 22          | 80                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 10             | off               | 87.50%   | 0.34 | 92%              |
| 23          | 120               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 10             | off               | 91.70%   | 0.15 | 94%              |
| 24          | 120               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | off               | 91.70%   | 0.14 | 96%              |
| 25          | 160               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | off               | 92.00%   | 0.14 | 96%              |
| 26          | 160               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | off               | 87.20%   | 0.34 | 88%              |
| 27          | 160               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 70     | 0.0005        | 10             | off               | 87.20%   | 0.14 | 88%              |
| 28          | 160               | MobileNetV2 | 96x96      | 0.35             | 4       | 0.1      | 20     | 0.0005        | 10             | off               | 100.00%  | 0.21 | 94%              |
| 29          | 160               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 10             | off               | 100.00%  | 0.03 | 98%              |
| 30          | 200               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 10             | off               | 100.00%  | 0.05 | 98%              |
| 31          | 200               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 20     | 0.0005        | 5              | off               | 100.00%  | 0.05 | 98%              |
| 32          | 200               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 90.00%   | 1.80 | 100%             |
| 33          | 160               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 94.10%   | 0.10 | 100%             |
| 34          | 120               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 92.30%   | 0.09 | 100%             |
| 35          | 80                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 90.00%   | 0.57 | 94%              |
| 36          | 40                | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 80.00%   | 0.65 | 80%              |
| 37          | 200               | MobileNetV2 | 96x96      | 0.35             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 94.70%   | 0.07 | 100%             |
| 38          | 200               | MobileNetV1 | 96x96      | 0.25             | 0       | 0.1      | 40     | 0.0005        | 10             | on                | 89.50%   | 0.29 | 64%              |
| 39          | 200               | MobileNetV1 | 96x96      | 0.25             | 0       | 0.1      | 60     | 0.0005        | 5              | on                | 90.00%   | 0.27 | 64%              |
| 40          | 200               | MobileNetV1 | 96x96      | 0.25             | 0       | 0.1      | 40     | 0.0005        | 5              | on                | 78.90%   | 0.35 | 51%              |
| 41          | 200               | MobileNetV1 | 96x96      | 0.2              | 0       | 0.1      | 100    | 0.0005        | 5              | on                | 100.00%  | 0.09 | 71%              |
| 42          | 200               | MobileNetV1 | 96x96      | 0.2              | 0       | 0.1      | 120    | 0.0005        | 5              | on                | 90.00%   | 0.24 | 79%              |
| 43          | 200               | MobileNetV1 | 96x96      | 0.2              | 0       | 0.1      | 160    | 0.0005        | 5              | on                | 90.00%   | 0.21 | 79%              |
| 44          | 200               | MobileNetV1 | 96x96      | 0.2              | 0       | 0.1      | 160    | 0.0005        | 5              | on                | 100.00%  | 0.04 | 96%              |
| 45          | 200               | MobileNetV2 | 96x96      | 0.1              | 0       | 0.1      | 160    | 0.0005        | 5              | on                | 100.00%  | 0.01 | 93%              |
| 46          | 200               | MobileNetV2 | 96x96      | 0.05             | 0       | 0.1      | 160    | 0.0005        | 5              | on                | 100.00%  | 0.00 | 93%              |
| 47          | 200               | MobileNetV1 | 96x96      | 0.1              | 0       | 0.1      | 160    | 0.0005        | 5              | on                | 100.00%  | 0.04 | 100%             |

## Results

The selected final model exhibits exceptional performance, achieving a 100% accuracy rate on the test dataset and a 96.67% accuracy rate on the validation set, despite the Edge Impulse website reporting 100% accuracy. Furthermore, it maintains a low inference time of 207 ms. Overall, I am extremely pleased with the model's performance, as it effectively meets my project objective.

![image](https://user-images.githubusercontent.com/73647889/232231896-f7d0ae15-701d-4e37-9eb4-6a047422b04e.png)

The selected model, MobileNetV1 96x96 0.1, exhibited different inferencing times when tested on various embedded devices. These findings emphasize that more powerful devices can substantially reduce inferencing time compared to less capable devices. In the context of self-checkout systems, faster inferencing times are preferable for a seamless user experience. However, the cost of the device should also be considered to balance the expenses associated with purchasing and replacing sensors against the performance improvements achieved.

| Embedded Device | Inferencing time (ms) |
| --- | --- |
| Arduino Nano 33 BLE Sense (Cortex-M4F 64MHz)| 207 |
| Arduino Nicla Vision (Cortex-M7 480MHz) | 16 |
| Arduino Portenta H7 (Cortex-M7 480MHz) | 16 |

## Reflection
A more systematic approach to testing could have been beneficial, as various transfer learning models were only explored in the 38th experiment and beyond due to preconceived notions about MobileNetV2's effectiveness. Testing different models earlier might have helped identify the best option sooner, allowing more time for fine-tuning. By tracking inferencing time, peak RAM usage, and flash usage from the beginning, it would be possible to better understand how different models and hyperparameters impact on-device performance and further reduce inference time.

More time should have been spent testing the model outside of the edge impulse transfer learning and model testing section. To increase the model's robustness, additional edge cases should have been considered to ensure accurate labelling. Scenarios not taken into account include instances where no apple is present, situations where the camera might misclassify objects e.g., an orange classified as an apple, or instances where the model might incorrectly classify a golden apple as a pink lady or granny smith.
  
## Future
* Develop a more extensive training and testing dataset to enhance model robustness and enable more accurate performance comparisons between models
* Explore a wider variety of transfer learning models
* Evaluate model performance on various embedded devices
* Experiment with different image resolutions
* Detect when no apple or object is present in the scene
* Incorporate a broader selection of fruits and vegetables
* Generate training and testing data that closely resemble a self-checkout environment for improved practicality
  

