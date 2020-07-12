## COVID-ResNet - COVID-19 Diagnosis from Chest X-Rays/CT scans using Deep Convolutional Neural Networks

I am not a medical professional and have no knowledge of COVID-19. Please do not use this model for anything other than a reference that can be built upon. This is by no means a production-ready solution.

Inspired by [COVID-Net](https://github.com/lindawangg/COVID-Net)

COVID-Net's design piqued my curiosity, as a binary classification was not used, but images were classified into three categories. Instead of classifying into three categories, 'normal', 'pneumonia', and 'COVID-19', if pneumonia was classified with normal, it should be possible to attain high accuracy. After experimenting with different architectures, ResNet-18 was found to be the most effective. Also included in this repository are MLP and ResNet50 implementations, however, only ResNet-18 has been tuned. All ResNet-50 and MLP parameters are arbitrary and should be tuned.

### Results Using ResNet-18
![confusion matrix](https://img.techpowerup.org/200712/index.png)

Precision - 98.75%

### Training
Instructions to create the dataset can be seen [here](https://github.com/Stanley-Zheng/COVID-ResNet/blob/master/makedataset.md)


### Further Reading and Sources
Paper - https://pubs.rsna.org/doi/10.1148/radiol.2020201874

Dataset - https://github.com/ieee8023/covid-chestxray-dataset

Dataset - https://github.com/agchung/Figure1-COVID-chestxray-dataset

Dataset - https://github.com/agchung/Actualmed-COVID-chestxray-dataset

Dataset - https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
