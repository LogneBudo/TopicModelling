using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace ModelTopcing
{
    internal class Program
    {
        public class TextData
        {
            public string Text { get; set; }
        }
        static void Main(string[] args)
        {
            var corpusFilePath = "corpus.txt";
            var stopwordsFilePath = "stopwords.txt";
            var ldaModelFilePath = "lda.zip";

            var stopwords = new HashSet<string>(File.ReadAllLines(stopwordsFilePath));
            var corpus = File.ReadAllLines(corpusFilePath);

            var mlContext = new MLContext();

            // Load the data
            var data = new List<TextData>();
            foreach (var line in corpus)
            {
                data.Add(new TextData { Text = line });
            }

            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Define the text processing pipeline
            var pipeline = mlContext.Transforms.Text.NormalizeText("NormalizedText", "Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceNgrams("Ngrams", "Tokens", ngramLength: 1, useAllLengths: false))
                .Append(mlContext.Transforms.Text.LatentDirichletAllocation("Features", "Ngrams", numberOfTopics: 5));

            // Train the model
            var model = pipeline.Fit(dataView);

            // Save the model
            mlContext.Model.Save(model, dataView.Schema, ldaModelFilePath);

            // Transform the data
            var transformedData = model.Transform(dataView);

            // Extract the LDA topics
            var ldaColumns = transformedData.GetColumn<float[]>("Features").ToArray();

            // Get the slot names (words) for the Ngrams column
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedData.Schema["Ngrams"].GetSlotNames(ref slotNames);
            var words = slotNames.DenseValues().Select(x => x.ToString()).ToArray();

            Console.WriteLine("Top words for each topic:");
            for (int topicIndex = 0; topicIndex < 5; topicIndex++)
            {
                var topWords = words
                    .Select((word, index) => new { Word = word, Weight = ldaColumns.Select(doc => doc[topicIndex]).Average() })
                    .OrderByDescending(x => x.Weight)
                    .Take(10)
                    .Select(x => x.Word);

                Console.WriteLine($"Topic {topicIndex}: {string.Join(", ", topWords)}");
            }

        }
    }
}
