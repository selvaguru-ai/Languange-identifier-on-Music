I would credit https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ (Valerio Velardo - The Sound of AI) youtube channel to help me build this project and https://www.appliedaicourse.com/ which according to me is a Bible that introduced me to the world of Machine Learning.

This simple model I have developed identifies what language the song is in by using simple audio signal processing techniques combined with a Multi-layer perceptron network.
For simplicity this is a binary classification that identifies whether a song is in English or Tamil this can be further extended to classify other languages also if trained with datasets accordingly.

I built this by creating a dataset with total of 40 songs (20 English and 20 Tamil) then trained my MLP on this to find the appropriate weights then tested to get an accuracy of 63.33% (This  accuracy can further be increased when Dropout layers are included which reduces overfitting or using other superior models such as CNN or RCNN.

Data- Preprocessing:- Using Fast Fourier Transform I calculated the MFCC values of every song by sampling each song into 10 parts. These MFCC values are the main components of the dataset. The songs are trained based on these MFCCs and the language of the song is determined using the same.
