using Azure;
using Azure.AI.OpenAI;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using OpenAI.Images;

namespace DallEWebApp.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        [BindProperty]
        public string Prompt { get; set; }
        public string ImageUrl { get; set; }

        public string RevisedPrompt { get; set; }

        public IndexModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
        }

        public void OnGet()
        {

        }

        public async Task<IActionResult> OnPostAsync()
        {
            string endpoint = "";
            string apiKey = "";
            AzureOpenAIClient azureClient = new(new Uri(endpoint), new AzureKeyCredential(apiKey));
            ImageClient imageClient = azureClient.GetImageClient("deployment-name ");//Name of the deployed DALL-E model

            var imageGenerationResult = await imageClient.GenerateImageAsync(Prompt,
                new ImageGenerationOptions
                {
                    Size = GeneratedImageSize.W1024xH1024

                });

            ImageUrl = imageGenerationResult.Value.ImageUri.ToString();
            RevisedPrompt = imageGenerationResult.Value.RevisedPrompt;

            return Page();
        }
    }
}
