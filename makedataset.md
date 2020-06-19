## Creating the Dataset

I am using a combination of 4 opensource COVID-19 datasets. 
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

To start, execute the following commands to clone the github repositories. Make sure your current directory is the clone of my repo.
`git clone https://github.com/ieee8023/covid-chestxray-dataset`
`git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset`
`git clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset`

The Kaggle dataset must be downloaded and unzipped to the same location. Navigate to where the Kaggle dataset was downloaded, and find the 'COVID-19' and 'NORMAL' folders. Create a 'VERIFICATION' file alongside them. 
Next, run `makedataset.py`. It will move all one of three folders in 'chest x-rays into the COVID-19 Radiography Database'. If it was a positive test, the image is in 'COVID-19', and if it was normal, the image is in 'NORMAL'. Certain images designated as verification have been placed in `verification.txt` and will be moved to the 'VERIFICATION' folder for testing of our neural network. Any images in 'VERIFICATION' will not be used for training. 
