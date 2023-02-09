<div align="center">

![](https://user-images.githubusercontent.com/55794407/204733431-d3bceacb-3830-410c-bd4c-04fa1173bc39.png)

</div>

Welcome to Hugging Face tutorials 

Hugging Face is an awesome platform to use and share NLP models. Whether you are an NLP practitioner or researcher, Hugging Face is a must-learn tool for your NLP projects. This repo contains Hugging Face tutorials 🤗. You can find notebooks, blog posts and videos here. Don't forget to follow us and star this repo ✨

## What is Hugging Face?

Hugging Face is a library that provides pre-trained language models, for NLP tasks such as text classification, sentiment analysis, and more. 

## What is Transformers?

Transformers allow you to use APIs and tools to easily download and train state-of-the-art pretrained models. With pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

📝 Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation. <br>
🖼️ Computer Vision: image classification, object detection, and segmentation. <br>
🗣️ Audio: automatic speech recognition and audio classification. <br>
🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering. <br>

## The Hugging Face Ecosystem

The Hugging Face ecosystem consists of four main components: the Hub, the Transformers and Diffusers libraries, the Inference API, their web-application builder Gradio.

### The Hub

Hugging Face Hub is a platform with models, datasets, and demo applications. The Hub works as a central place for all things related to machine learning.  Now let's take a look at what's inside the Hub.

*Models*: In this Hub, you can discover and use dozens of thousands of open-source ML models shared by the community. The Hub makes it easy to explore new models and get started with NLP projects.

*Datasets*: The Hub contains 5,000 datasets in over 100 languages. You can use these datasets for a broad range of tasks across NLP, Computer Vision, and Audio. The hub makes it easy to find, download, and upload datasets.

*Space*: Space allows you to deploy your machine learning demo apps. These apps help you build your ML portfolio, showcase your projects, and work collaboratively with other people. Hugging Face lets you use Gradio and Streamlit libraries. You can also create static Spaces which are simple HTML, CSS and JavaScript page within a Space.

### The Transformer and Diffusers Libraries

#### The Transformer Library

Transformers is an open source library, provides APIs and tools. So, you can easily download, and train state-of-the-art pretrained models. It is built on the PyTorch framework. You can also use it with TensorFlow and JAX. This provides the flexibility to use a different framework at each stage of a model. For example, you can train a model, in three lines of code in one framework, and load it for inference in another. It supports most tasks from NLP to Audio, CV and even Multimodal tasks.

#### The Diffusers library

The Diffusers library is a new addition to the Hugging Face ecosystem.  It provides a way to easily share, version, and reproduce pre-trained diffusion models, for computer vision and audio tasks.  This library focuses on diffusion models, and in particular Stable Diffusion, which has been open source since August 2022.  The Diffusers library allows you, to use stable diffusion in an easy way.

## The Inference API

Let's say you want to put a model in Hugging Face into a production environment. You can use the Hugging Face Interface API for this. 

## Gradio

Let's say you built a model and want to move it into a production environment. To do this, you can use Gradio library on Hugging Face. This library allows you, to build a web application in a minute. Gradio has great documentation. You can easly learn it using this documantaion.

## Quick tour

You can make predictions with the pipeline API. Pipelines group together a pre-trained model with the preprocessing that was used during that model's training. Let's take a look at sentiment analysis.

```
>>> from transformers import pipeline
>>> classifier = pipeline("text-classification")
>>> classifier('I am very excited to cover the pipeline to the transformers tutorial.')
[{'label': 'POSITIVE', 'score': 0.9987786412239075}]
```

## My Youtube Videos

- [Getting Started with HuggingFace](https://youtu.be/ir-_Ds_d8k4)
- [Hugging Face Pipelines](https://youtu.be/z-w4d7K010g)


