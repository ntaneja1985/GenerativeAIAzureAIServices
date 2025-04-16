# AI Agents with .NET Aspire
- Refer to this link: https://github.com/microsoft/AI_Agents_Hackathon/discussions/58
- Agent is a Semi-autonomous software that can be given a goal and will work to achieve that goal without you knowing in advance exactly how it's going to do that or what steps it's going to take.
- ![alt text](image-573.png)
- ![alt text](image-574.png)
- ![alt text](image-575.png)
- Semantic Kernel is an orchestration middleware that lets us add AI to our applications
- It is built specifically for enterprise app developers
- Extensions AI and Semantic Kernel both help to add AI to our code
- ![alt text](image-576.png)
- We can use Semantic Kernel even to talk to local models like Ollama or ONNX
- Semantic Kernel is very extensible.
- We can use Semantic Kernel to talk to Vector Databases like Azure AI Search
- We can also use Open Telemetry with this
  
### AI Agents in Action
- A conversation with an Agent is called a Thread
- ![alt text](image-577.png)
- ![alt text](image-578.png)
- Refer to this github repo: https://github.com/Azure-Samples/aspire-semantic-kernel-creative-writer
- ![alt text](image-579.png)

### Where does .NET Aspire fit in
- ![alt text](image-580.png)
- ![alt text](image-581.png)
- ![alt text](image-582.png)
- ![alt text](image-583.png)
- ![alt text](image-584.png)
- ![alt text](image-585.png)
- ![alt text](image-586.png)
- ![alt text](image-587.png)
- ![alt text](image-588.png)
- ![alt text](image-589.png)
- .NET Aspire sounds a lot like docker compose
- The builder.AddDistributedApplication method is part of the Aspire.Hosting framework in .NET, used for creating and managing distributed applications. It allows developers to define resources, services, and configurations for distributed systems
```c#
var builder = DistributedApplication.CreateBuilder(args);
var cache = builder.AddRedis("cache");
var inventoryDatabase = builder.AddPostgres("postgres").AddDatabase("inventory");
builder.AddProject<Projects.InventoryService>("inventoryservice")
       .WithReference(cache)
       .WithReference(inventoryDatabase);
builder.Build().Run();

```
- ![alt text](image-590.png)
- ![alt text](image-591.png)
- ![alt text](image-592.png)
- ![alt text](image-593.png)
- This command will give Azure Bicep Files
- ![alt text](image-594.png)
- ![alt text](image-595.png)
- ![alt text](image-596.png)
- Bicep is a type of file which we can use to deploy to Azure
- It will try to deploy everything to Azure
- ![alt text](image-597.png)
- It is connected to Azure AI Search Instance
- We can define multiple agents
- We can have expenses agent or sender agents
- ![alt text](image-598.png)
- We can connect Agents to external data sources
- They can even connect to Azure Functions
- ![alt text](image-599.png)
- ![alt text](image-600.png)
- ![alt text](image-601.png)
- ![alt text](image-602.png)
- ![alt text](image-603.png)
- ![alt text](image-604.png)
- ![alt text](image-605.png)
- ![alt text](image-606.png)
- ![alt text](image-607.png)
- ![alt text](image-608.png)
- ![alt text](image-609.png)
- 