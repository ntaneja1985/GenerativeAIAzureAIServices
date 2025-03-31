# Generative AI with Azure AI Services
- ![alt text](image.png)
- Machine Learning is a form of AI that is able to learn without explicit programming by human
- Some ML algorithms are specialized in training themselves to detect patterns; this is called deep learning.
- ![alt text](image-1.png)
- Generative AI is AI capable of creating new content using models trained on existing data.
- It generates text,images, audio and video
- It is powered by advanced machine learning models such as transformers, GANs(Generative Adversarial Networks) and others
- Examples of GenAI are ChatGPT(for text), DALL-E(for images) and MusicLM(for audio)
- GenAI is not fully autonomous. It relies on training data and models built by humans. Cant think or reason independently.
- AI Models can inherit biases from the data they are trained on.
- GenAI content may not always be accurate or reliable. 
- AI is a tool meant to augment human capabilities, not replace them entirely.

## Ethical implications of AI
- AI has potential to unlock all kinds of opportunities for businesses, governments and society.
- It should promote non-discrimination and fairness.
- It should not violate human rights.
- It must minimize environmental impact, reduce carbon footprint
- It should follow privacy and data protection laws
- It should be accountable.


## Machine Learning Basics
- Machine Learning is a form of AI that is able to learn without explicit programming by human
- Some ML algorithms are specialized in training themselves to detect patterns; this is called deep learning.
- 2 types of machine learning: Supervised learning(map points between input and output) and unsupervised learning(discover patterns and relationships using unlabelled data)
- Unsupervised learning tries to find clusters of data
- Additional type of ML is Semi-supervised learning(uses both labelled and unlabelled data)
- We also have reinforcement learning which mimics the trial and error learning process that humans use to achieve their goals e.g autonomous vehicles.
- ![alt text](image-2.png)
- ![alt text](image-3.png)
- ![alt text](image-4.png)

## Understanding Neural Networks
- They are computational models designed to recognize patterns and make decisions based on data. 
- ![alt text](image-5.png)
- There are two main processes for a neural network.
- There is the forward propagation, where data moves from the input layer through the hidden layers to the output layer and each neuron.
- So each circle here in that hidden layer that represents a neuron, it applies some mathematical function to the input.
- That's where that weighting comes.
- So the higher the mathematical, the output of it or whatever parameters there are to determine what's a good, um, outcome of that mathematical operation or not, it will give it more or less precedence towards a final response.
- We have back propagation where the network adjusts its weights based on the error of its predictions, improving accuracy over time. So that is why you'll see generative AI getting smarter each time as well. And sometimes they ask for feedback.
- ![alt text](image-6.png)

## Introduction to ML.NET
- Ml.net is an open source, cross-platform machine learning framework that is designed specifically for .Net developers, it allows us to build, train and deploy machine learning models using C# or F#.
- So with this, we also get AutoML capabilities which help us to automatically build and tune models.
- We also have Azure AI Services: Set of cloud based artificial intelligence tools and API that allow us to build AI powered applications without needing in-depth AI expertise.
- Another thing is Semantic Kernel which is an open source framework which allows us to integrate LLMs in our applications through this library.
- We have ONNX(Open Neural Network Exchange): Open source format for representing ML Models.
- We have PyTorch from Facebook and TensorFlow which comes from Google.
- We have CnTk(Microsoft Cognitive Toolkit) from Microsoft for deep learning. Helps to build applications optimized for speech, image recognition and text.
- ![alt text](image-7.png)
- ![alt text](image-8.png)
- ![alt text](image-10.png)
- ![alt text](image-11.png)
- ![alt text](image-12.png)
- ![alt text](image-13.png)
- ![alt text](image-14.png)
- We have a concept of overfitting, which basically means that the model has been trained on the data a bit too well such that newer data seems like rubbish to it, so we don't want it to be too familiar and too confident with just the data set it got.
```c#
//Use the following code to interact with the Predictive Model and predict values using it
//Load sample data
var sampleData = new PredictiveModel.ModelInput()
{
    UDI = 2F,
    Product_ID = @"L47181",
    Air_temperature = 298.2F,
    Process_temperature = 308.7F,
    Rotational_speed = 1408F,
    Torque = 46.3F,
    Tool_wear = 3F,
};

//Load model and predict output
var result = PredictiveModel.Predict(sampleData);


```

- We can add a machine learning model to our project like this
- ![alt text](image-15.png)
- Once we train our model with the sample dataset we can use it to predict values like this:
```c#
using MLSampleAppConsole;

//Prompt the user for input values
//Build the input model from values
PredictiveModel.ModelInput modelInput = new()
{
    UDI = 2F,
    Product_ID = @"L47181",
    Air_temperature = 298.2F,
    Process_temperature = 308.7F,
    Rotational_speed = 1408F,
    Torque = 46.3F,
    Tool_wear = 3F,
};

Console.WriteLine("Comparing actual Machine_failure with predicted Machine_failure from sample data...\n\n");


//Passing input values into predictive model
var scoresWithLabel = PredictiveModel.PredictAllLabels(modelInput);

Console.WriteLine($"{"Class",-40}{"Score",-20}");
Console.WriteLine($"{"-----",-40}{"-----",-20}");

foreach (var score in scoresWithLabel)
{
    Console.WriteLine($"{score.Key,-40}{score.Value,-20}");
}
Console.WriteLine($"{"-----",-40}{"-----",-20}");
var modelOutput = PredictiveModel.Predict(modelInput);
Console.WriteLine("Single Prediction Result");
Console.WriteLine($"UDI: {modelOutput.UDI}");
Console.WriteLine($"Product_Id: {modelOutput.Product_ID}");
Console.WriteLine($"Air_Temperature: {modelOutput.Air_temperature}");
Console.WriteLine($"Tool_wear: {modelOutput.Tool_wear}");
Console.WriteLine($"Machine_failure: {modelOutput.Machine_failure}");


Console.WriteLine("------------Done------------");
Console.ReadKey();


```

## Consuming a model from .NET API
- We can use the following code to consume the PredictiveModel like this
```c#
// This file was auto-generated by ML.NET Model Builder. 
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;
using Microsoft.OpenApi.Models;
using Microsoft.ML.Data;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;

// Configure app
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddPredictionEnginePool<PredictiveModel.ModelInput, PredictiveModel.ModelOutput>()
    .FromFile("PredictiveModel.mlnet");

builder.Services.AddEndpointsApiExplorer();

builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "My API", Description = "Docs for my API", Version = "v1" });
});
var app = builder.Build();

app.UseSwagger();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "My API V1");
    });
}


// Define prediction route & handler
app.MapPost("/predict",
    async (PredictionEnginePool<PredictiveModel.ModelInput, PredictiveModel.ModelOutput> predictionEnginePool, PredictiveModel.ModelInput input) =>
        await Task.FromResult(predictionEnginePool.Predict(input)));

// Run app
app.Run();


```

