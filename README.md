# TopicModelling

## Overview
TopicModelling is a .NET Framework 4.8 application that uses machine learning to identify topics in a corpus of text documents. It leverages the Microsoft.ML library to perform Latent Dirichlet Allocation (LDA) for topic modeling.

## Features
- **Text Normalization**: Cleans and normalizes text data.
- **Tokenization**: Splits text into individual words.
- **Stop Words Removal**: Removes common stop words.
- **N-grams Generation**: Produces n-grams from tokens.
- **Latent Dirichlet Allocation**: Identifies topics in the text data.

## Getting Started

### Prerequisites
- .NET Framework 4.8
- Visual Studio

### Installation
1. Clone the repository:
    
git clone https://github.com/lognebudo/TopicModelling.git

2. Open the solution in Visual Studio.

3. Install the `Microsoft.ML` package via NuGet Package Manager Console:

```bash
Install-Package Microsoft.ML
```

### Usage
1. Prepare your corpus and stopwords files:
    - `corpus.txt`: Contains the text data, one document per line.
    - `stopwords.txt`: Contains stop words, one per line.

2. Update the file paths in `Program.cs`:

```csharp
var corpusFilePath = "path/to/your/corpus.txt";
var stopwordsFilePath = "path/to/your/stopwords.txt";
var ldaModelFilePath = "path/to/save/lda.zip";
```

3. Build and run the project in Visual Studio.

### Example Output

## Project Structure

TopicModelling/ 
├── Program.cs 
├── corpus.txt 
├── stopwords.txt 
└── lda.zip

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [Microsoft.ML](https://github.com/dotnet/machinelearning) for the machine learning library.
- [Visual Studio](https://visualstudio.microsoft.com/) for the development environment.

## Screenshots
![Visual Studio](https://user-images.githubusercontent.com/yourusername/visualstudio.png)