# Go Swear Words Logistic Regression

This project implements a logistic regression model to detect swear words in text. The model is trained using a predefined vocabulary and can predict whether a given word is considered profane.

## Profanity List

The profanity list used in this project is sourced from:
[Surge AI Profanity List](https://github.com/surge-ai/profanity)

## Installation

To install and run this project, follow these steps:

1. Clone the repository:
2. Install the required dependencies:
   ```sh
   go mod tidy
   ```

## Usage

To train the model and run predictions, follow these steps:

1. Prepare your training data and test cases within the code.

2. Run the tests to ensure everything is working correctly:

   ```sh
   go test
   ```

3. Use the logistic regression model in your application:

   ```go
   package main

   import (
       "fmt"
   )

   func main() {
       lr := &LogisticRegression{}

       // Define your vocabulary
       vocabulary := map[string]int{
           "hello":    0,
           "world":    1,
           "goodbye":  2,
       }

       // Define your training data
       trainingData := []struct {
           text  string
           label float64
       }{
           {"hello world", 0},
           {"goodbye world", 1},
       }

       // Prepare training samples and labels
       var samples [][]float64
       var labels []float64
       for _, data := range trainingData {
           sample := make([]float64, len(vocabulary))
           words := strings.Fields(data.text)
           for _, word := range words {
               if index, exists := vocabulary[strings.ToLower(word)]; exists {
                   sample[index]++
               }
           }
           samples = append(samples, sample)
           labels = append(labels, data.label)
       }

       // Train the model
       lr.train(samples, labels)

       // Predict a new sample
       newSample := make([]float64, len(vocabulary))
       newSample[vocabulary["hello"]]++
       prediction := lr.predict(newSample)
       fmt.Printf("Prediction: %f\n", prediction)
   }
   ```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## Tests

The project is tested by one integration test that check the entire flow of the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
