# What is Hugging Face?

AI has become more, and more part of our daily lives, in recent years. AI tools such as ChatGPT and DALL-E, are developed with deep learning techniques. Deep learning is a subfield of AI that aims to extract knowledge, from data through complex neural networks.

Deep learning analysis is difficult. Because it requires a lot of mathematical calculations. Fortunately, deep learning analysis can be perform more easily with the libraries such as TensorFlow and PyTorch that have been developed recently. Building a deep learning model takes both money and time. Fortunately, when performing a task, we can reuse a previously trained model for that task. This technique is called transfer learning.

With the increasing use of transfer learning, some problems have emerged. Let's take a look at these problem. These pre-trained models were trained using different libraries, such as TensorFlow and PyTorch. It needed to standardize these differences. Since the models were also very large, it should load the weights of these models from the server. To overcome these problems, the Hugging Face library was developed. Let's get started with what Hugging Face.

##  What is Hugging Face?

If you're in the field of Natural Language Processing, you've probably heard about Hugging Face. But what is it exactly and why is it so important? Hugging Face is a library. That provides pre-trained language models, for NLP tasks such as text classification, sentiment analysis, and more. These models are based on deep learning algorithms and have been fine-tuned for specific NLP tasks, making it easy to get started with NLP. 

The best part is, you don't have to train a model from scratch. Just load a pre-trained model, and fine-tune it to your specific task and with Hugging Face, it's as simple as that.

Another reason why Hugging Face is so popular is its ease of use. It has a simple, and user-friendly interface. So this making it easy for developers, to quickly get started with NLP. 

And let's not forget about the amazing community support. The Hugging Face community is active and supportive. The community also contributes to the growth of the library by sharing pre-trained models. With Hugging Face you can easily tackle NLP tasks.

Okay. We've learned what Hugging Face is. Let’s now go ahead and review the features of this platform.

## Hugging Face Features

Hugging Face is an awesome platform that provides a suite of powerful tools for various tasks. Let me explain the main components of Hugging Face. The main features of the platform are the Hub, Transformers and Diffusers libraries, the Inference API, their web-application builder Gradio. In this section, we're going to discuss these key features of Hugging Face. Let's start with the Hub.

## The Hub

The first feature we'll discover at is the Hub. Hugging Face Hub is a platform with models, datasets, and demo applications. The Hub works as a central place for all things related to machine learning.  Now let's take a look at what's inside the Hub.

### Models

In this Hub, you can discover and use dozens of thousands of open-source ML models shared by the community. The Hub makes it easy to explore new models and get started with NLP projects.

### Datasets

The Hub contains 5,000 datasets in over 100 languages. You can use these datasets for a broad range of tasks across NLP, Computer Vision, and Audio. The hub makes it easy to find, download, and upload datasets.

### Space

Let's now talk about space. Space allows you to deploy your machine learning demo apps. These apps help you build your ML portfolio, showcase your projects, and work collaboratively with other people.

Hugging Face lets you use Gradio and Streamlit libraries. You can also create static Spaces which are simple HTML, CSS and JavaScript page within a Space. Let me give you the good news these features in the hub are hosted as git repositories. So you can version models, datasets and demo apps and track changes in them. You can also contribute to other researchers' projects. Let's move on to have a look at libraries in Hugging Face.

## The Transformer and Diffusers Libraries

In this section, we're going to handle the Transformer and Diffusers Libraries. Let's start with the Transformer library. 

### The Transformer Library

Transformers is an open source library, provides APIs and tools. So, you can easily download, and train state-of-the-art pretrained models. Transformers is built on the PyTorch framework. You can also use it with TensorFlow and JAX. This provides the flexibility to use a different framework at each stage of a model. For example, you can train a model, in three lines of code in one framework and load it for inference in another. It supports most tasks from NLP to Audio, CV and even Multimodal tasks. Okay, we've learned Transformers library. Let's go ahead and discuss another library in Hugging Face.

### The Diffusers library

The Diffusers library is a new addition to the Hugging Face ecosystem. It provides a way to easily share, version, and reproduce pre-trained diffusion models, for computer vision and audio tasks. This library focuses on diffusion models, and in particular Stable Diffusion, which has been open source since August 2022. The Diffusers library allows you, to use stable diffusion in an easy way. Nice, let's go ahead, and take a look at the Inference API.

### The Inference API

Let's say you want to put a model in Hugging Face into a production environment. You can use the Hugging Face Interface API for this. Let's take a look at how to do this. The first step is to choose, which model you are going to run.

Let’s use gpt2 as an example. 

- Go to [the page of GPT 2](https://huggingface.co/gpt2). 
- Let's click the deploy button.
- Here is the API url for this model.

```
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})
```

- Type your API token into the Header variable. 
- You can predict with the query function. 

The prediction is shown for this data. Isn't it easy? In a nutshell, the Inference API allows you, to integrate NLP models into your existing applications, without having to write complex code. Cool, let's go ahead and take a look at how to use Gradio in Hugging Face.

### Gradio

Let's say you built a model and want to move it into a production environment. To do this, you can use Gradio library on Hugging Face. This library allows you, to build a web application in a minute. Gradio has great documentation. You can easly learn it using this documantaion. In a nutshell, With Gradio, you can create, and share your own NLP models with others, without having to be an expert in web development. If you're more interested in HuggingFace, you can check out the Hugging Face documentation. Here you can find great courses to learn Hugging Face.

## Conclusion

In conclusion, Hugging Face is an awesome powerful platform for NLP. Hugging Face provides great features to use, and share NLP models. Whether you are an NLP practitioner or researcher, Hugging Face is a must-learn tool for your NLP projects. That's it. Thanks for reading. Bye for now.

## Resources

- https://medium.com/mlearning-ai/huggingface-the-netflix-of-machine-learning-8fe93df8ccb
- https://huggingface.co/docs/transformers/quicktour
