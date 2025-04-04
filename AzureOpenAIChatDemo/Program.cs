//Create a Chatbot to interact with the deployed model

using System;
using System.Threading.Tasks;
using Azure;
using Azure.AI.OpenAI;
using OpenAI.Chat;


class Program
{
    static async Task Main()
    {
        // Configure Azure OpenAI details
        string endpoint = "https://YOUR_AZURE_OPENAI_ENDPOINT.openai.azure.com/";
        string apiKey = "YOUR_AZURE_OPENAI_API_KEY";
        string deploymentName = "gpt-35-turbo-16k";  // Example: "gpt-4-turbo"

        AzureKeyCredential credential = new AzureKeyCredential(apiKey);
        // Create OpenAI client
        AzureOpenAIClient azureClient = new (new Uri(endpoint), credential);
        ChatClient chatClient = azureClient.GetChatClient(deploymentName);
        string systemMessage = "You are a senior developer with 10 years of .NET development experience. Have a good sense of humor";

        Console.WriteLine("Azure OpenAI ChatBot - Type 'exit' to quit.");
        while (true)
        {
            Console.Write("You: ");
            string userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput) || userInput.ToLower() == "exit")
                break;

            // Send user input to Azure Open AI
            ChatCompletion completion = await chatClient.CompleteChatAsync(
                new SystemChatMessage(systemMessage),
                new UserChatMessage("Hi, can you help me"),
                new AssistantChatMessage("Yes, of course how can i mentor you today"),
                new UserChatMessage(userInput)
                );

            //Output the response content
            Console.WriteLine($"AI: {completion.Content[0].Text}\n");
        }
    }
}



