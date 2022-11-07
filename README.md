<div align="center">

![](https://camo.githubusercontent.com/b253a30b83a0724f3f74f3f58236fb49ced8d7b27cb15835c9978b54e444ab08/68747470733a2f2f68756767696e67666163652e636f2f64617461736574732f68756767696e67666163652f646f63756d656e746174696f6e2d696d616765732f7265736f6c76652f6d61696e2f7472616e73666f726d6572735f6c6f676f5f6e616d652e706e67)

</div>

In this repo, I'll talk about natural language processing with transformers using Hugging Face framework 🤗. You can find tutorials and notebooks in this repo. Don't forget to star and like this repo.

## 📌 What is Transformers?

Transformers allow you to use APIs and tools to easily download and train state-of-the-art pretrained models. With pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

📝 Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation. <br>
🖼️ Computer Vision: image classification, object detection, and segmentation. <br>
🗣️ Audio: automatic speech recognition and audio classification. <br>
🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering. <br>

✨ You can use Transformers with PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

## 📌 Quick tour

You can make predictions with the pipeline API. Pipelines group together a pre-trained model with the preprocessing that was used during that model's training. Let's take a look at sentiment analysis.

```
>>> from transformers import pipeline
>>> classifier = pipeline("text-classification")
>>> classifier('I am very excited to cover the pipeline to the transformers tutorial.')
[{'label': 'POSITIVE', 'score': 0.9987786412239075}]
```
