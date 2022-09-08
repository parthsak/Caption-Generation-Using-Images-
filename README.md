# Caption-Generation-Using-Images-

Describing the content of an image is one of the fundamental tasks that ultimately connects NLP and Computer Vision. It is inspired by works in the field of machine translation and object detection. Studying the region between visuals and language could point to a deeper vein of advancement that, once discovered, could lead to more advanced machines. It may be beneficial to retrieve portions of images which allows us to represent visual or image-based content in new ways. We want to accomplish this goal by using a model based on current advancements in computer vision and machine translation.

# Overview 
Architecture basically consists of encoder-decoder Neural Network, where encoder extracts feature from the image and decoder interprets those features to produce sentence.

Encoder consists of a pre-trained Inception-V3 model (minus the last Fully Connected layers) appended by a custom Fully Connected layer. Decoder consists of GRU along with visual-attention. Visual attention helps GRU in focusing on relevant image features for the prediction of a particular word in word sequence (sentence)

For word embedding, a custom word2vec model was trained and it was then intersected with Google's pre-trained word2vec model.

Flickr30 Dataset was used for creating train and test sets.

# Results 
![index](https://user-images.githubusercontent.com/65400703/189194792-e9eead0e-be51-455a-8cd6-616a4c682035.png)
![index1](https://user-images.githubusercontent.com/65400703/189194823-70d9cc87-a858-47e5-acbd-fdf2d1457507.png)
![index2](https://user-images.githubusercontent.com/65400703/189194843-858b9212-762a-419e-9570-9c52f349a206.png)
![index3](https://user-images.githubusercontent.com/65400703/189194869-8d0a0852-a5b9-452b-a94f-4688099c85b7.png)
