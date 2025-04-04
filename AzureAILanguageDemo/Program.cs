using Azure;
using Azure.AI.TextAnalytics;

string key = "MyKey";
string endpoint = "https://az-language-nish.cognitiveservices.azure.com/";


AzureKeyCredential azureKeyCredential = new AzureKeyCredential(key);
Uri languageEndpoint = new Uri(endpoint);

var client = new TextAnalyticsClient(languageEndpoint,azureKeyCredential);

var sentences = new List<string>
{
    "I love the new Italian restaurant. THe pastaa was very good",
    "We need to schedule a meeting to discuss the Q3 projections and finalize the budget"
};


//Pass a single line of text
DocumentSentiment sentenceToAnalyze = client.AnalyzeSentiment(sentences[0]);

//Pass a collection of text
// Summary:
//     Include Opinion Mining = Whether to mine the opinion of a sentence and conduct more granular analysis
//     around the targets of a product or service (also known as Aspect-Based sentiment
//     analysis). If set to true, the returned Azure.AI.TextAnalytics.SentenceSentiment.Opinions
//     will contain the result of this analysis.
AnalyzeSentimentResultCollection sentencesToAnalyze = client.AnalyzeSentimentBatch(sentences,
    options: new AnalyzeSentimentOptions() {IncludeOpinionMining = true });

foreach (AnalyzeSentimentResult sentence in sentencesToAnalyze)
{
    Console.WriteLine($"Sentence Sentiment:{sentence.DocumentSentiment.Sentiment}");

    Console.WriteLine($"Positive:{sentence.DocumentSentiment.ConfidenceScores.Positive}");

    Console.WriteLine($"Neutral:{sentence.DocumentSentiment.ConfidenceScores.Neutral}");

    Console.WriteLine($"Negative:{sentence.DocumentSentiment.ConfidenceScores.Negative}");

    foreach(var sentenceSentiment in sentence.DocumentSentiment.Sentences)
    {
        Console.WriteLine($"Sentence -Aspect: {sentenceSentiment.Text}");

        foreach(var sentenceOpinion in sentenceSentiment.Opinions)
        {
            Console.WriteLine($"Sentence Opinion: {sentenceOpinion.Target.Sentiment}");

            foreach(AssessmentSentiment assessment in sentenceOpinion.Assessments)
            {
                Console.WriteLine($"Related assessment: {assessment.Text}");
            }
        }

    }
}


