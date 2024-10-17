using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ModelTopicing
{
    internal class Program
    {
        // Class to hold the text data
        public class TextData
        {
            public string Text { get; set; }
        }

        static void Main(string[] args)
        {
            // File paths for the corpus, stopwords, and the LDA model
            var corpusFilePath = "corpus.txt";
            var stopwordsFilePath = "stopwords.txt";
            var ldaModelFilePath = "lda.zip";

            // Load stopwords from the file into a HashSet
            var stopwords = new HashSet<string>(File.ReadAllLines(stopwordsFilePath));
            // Load the corpus from the file into an array of strings
            var corpus = File.ReadAllLines(corpusFilePath);

            // Initialize MLContext
            var mlContext = new MLContext();

            // Load the data into a list of TextData objects
            var data = new List<TextData>();
            foreach (var line in corpus)
            {
                data.Add(new TextData { Text = line });
            }

            // Convert the list of TextData objects into an IDataView
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Define the text processing pipeline
            var pipeline = mlContext.Transforms.Text.NormalizeText("NormalizedText", "Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Tokens"))
                .Append(mlContext.Transforms.Text.ProduceNgrams("Ngrams", "Tokens", ngramLength: 1, useAllLengths: false))
                .Append(mlContext.Transforms.Text.LatentDirichletAllocation("Features", "Ngrams", numberOfTopics: 5));

            // Train the model using the pipeline
            var model = pipeline.Fit(dataView);

            // Save the trained model to a file
            mlContext.Model.Save(model, dataView.Schema, ldaModelFilePath);

            // Transform the data using the trained model
            var transformedData = model.Transform(dataView);

            // Extract the LDA topics from the transformed data
            var ldaColumns = transformedData.GetColumn<float[]>("Features").ToArray();

            // Get the slot names (words) for the Ngrams column
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedData.Schema["Ngrams"].GetSlotNames(ref slotNames);
            var words = slotNames.DenseValues().Select(x => x.ToString()).ToArray();

            // Print the top words for each topic
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