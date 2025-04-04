using Azure;
using Azure.AI.OpenAI;
using Azure.AI.OpenAI.Chat;
using OpenAI.Chat;

# pragma warning disable AOAI001

string endpoint = "";
string key = "";
string azureSearchEndpoint = "https://ai-search-nishant.search.windows.net";
string azureSearchKey = "";
string azureSearchIndex = "azuresearchindex";

AzureKeyCredential credential = new AzureKeyCredential(key);
AzureOpenAIClient azureClient = new AzureOpenAIClient(new Uri(endpoint), credential);
ChatClient chatClient = azureClient.GetChatClient("gpt-35-turbo-16k");

ChatCompletionOptions chatOptions = new();

//Configure the data source that we will use to answer the question
//In this case, we are using Azure Search as the data source
//Note that this helps to us to do Retrieval Augmented Generation (RAG) using Azure Search
chatOptions.AddDataSource(new AzureSearchChatDataSource()
{
    Endpoint = new Uri(azureSearchEndpoint),
    IndexName = azureSearchIndex,
    Authentication = DataSourceAuthentication.FromApiKey(azureSearchKey),
});

Console.WriteLine("Enter your question(type 'quit' to exit): ");
string userPrompt = Console.ReadLine() ?? string.Empty;

ChatCompletion completion = await chatClient.CompleteChatAsync(
    //Collection of User Chat Messages
    [
    new UserChatMessage(userPrompt)
    ],
    chatOptions);

ChatMessageContext chatContext = completion.GetMessageContext();
if(chatContext?.Intent is not null)
{
    Console.WriteLine($"Intent: {chatContext.Intent}");
}

//This is effective used to provide reference of the ebooks we had uploaded and vectorized earlier
foreach(ChatCitation citation in chatContext?.Citations ?? Enumerable.Empty<ChatCitation>())
{
    
    Console.WriteLine($"Citation: {citation.Content}");
    Console.WriteLine($"Citation URL: {citation.Url}");
}

