# human-transposons-enhancer-predicition
This model was tested on Google Colab Pro with A100 GPU

Transposable elements (TEs) are mobile genetic elements that constitute a significant portion of the human genome. Once considered "junk DNA," TEs are now recognized as key players in genome evolution and function. One of the most intriguing roles of TEs is their contribution to the regulatory landscape of the genome, particularly as enhancers. Enhancers are short DNA sequences that increase the transcription of genes, playing a crucial role in the spatial and temporal regulation of gene expression.

TE-derived enhancers have been implicated in a variety of biological processes, from embryonic development to the adaptive immune response. They have also been linked to disease when their regulatory functions are dysregulated. Despite their importance, our understanding of TE-derived enhancers is still in its infancy. This is partly due to the sheer diversity and abundance of TEs in the human genome, as well as their complex interactions with the surrounding genomic landscape.

Deep learning, a subset of machine learning, has emerged as a powerful tool for analyzing large and complex biological datasets. Deep learning models, such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), have been successfully applied to predict enhancers based on DNA sequence alone. However, most of these studies have focused on generic enhancers, without considering their origin.

The application of deep learning to the study of TE-derived enhancers represents a promising avenue for future research. By training models on datasets of known TE-derived enhancers, we can begin to unravel the sequence features that distinguish these elements from other parts of the genome. Furthermore, by including all TE subclasses in our analyses, we can gain a more comprehensive understanding of the diverse roles of TEs in gene regulation.

In conclusion, the intersection of TE biology and deep learning offers exciting opportunities for advancing our understanding of genome regulation. The characterization of TE-derived enhancers using deep learning not only has the potential to shed light on the functional importance of TEs but also to provide new insights into the genomic basis of human disease.

This project involves the use of a 1D Convolutional Neural Network (CNN) to distinguish between enhancer and non-enhancer sequences derived from transposable elements (TEs) in the human genome. The model is trained on experimentally derived enhancer sequences and transposon-derived sequences, which are preprocessed and one-hot encoded for input into the model.

The 1D CNN model architecture includes two convolutional layers, each followed by a max pooling layer, and a global max pooling layer. This is followed by a dense layer, a dropout layer for regularization, and a final dense layer with a sigmoid activation function for binary classification. The model is compiled with the Adam optimizer and binary cross-entropy loss function, suitable for the binary classification task.

The model is trained using a generator function to handle the large dataset, which yields batches of padded sequences and corresponding labels. The model is trained for 50 epochs, and the trained model is saved for later use.

The saved model is then loaded and used to predict the class (enhancer or non-enhancer) of unknown transposon sequences. The model outputs the probability of each sequence being an enhancer, and these probabilities are saved to a text file. Sequences are classified as enhancers if the predicted probability is greater than a specified threshold (0.5 in this case).

Finally, the sequences are ranked based on their predicted probabilities, and the top N sequences are selected as strong and weak enhancers. This information can be used to further investigate the role of specific TEs in gene regulation.

This project demonstrates the potential of deep learning, specifically 1D CNNs, in genomic research and the study of transposable elements.
