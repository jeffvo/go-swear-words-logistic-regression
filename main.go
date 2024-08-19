package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

type LogisticRegression struct {
    weights []float64
}

func NewLogisticRegression(inputSize int) *LogisticRegression {
    return &LogisticRegression{
        weights: make([]float64, inputSize),
    }
}

func sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

func computeLoss(yTrue, yPred float64) float64 {
    if yTrue == 1 {
        return -math.Log(yPred)
    }
    return -math.Log(1 - yPred)
}

func (lr *LogisticRegression) train(X [][]float64, y []float64, learningRate float64, epochs int, done chan bool) {
    for epoch := 0; epoch < epochs; epoch++ {
        var wg sync.WaitGroup
        totalLoss := 0.0
        for i := range X {
            wg.Add(1)
            go func(i int) {
                defer wg.Done()
                prediction := lr.predict(X[i])
                error := y[i] - prediction
                loss := computeLoss(y[i], prediction)
                totalLoss += loss
                for j := range lr.weights {
                    lr.weights[j] += learningRate * error * X[i][j]
                }
            }(i)
        }
        wg.Wait()
    }
    done <- true
}

func (lr *LogisticRegression) predict(features []float64) float64 {
    var sum float64
    for i := range features {
        sum += lr.weights[i] * features[i]
    }
    return sigmoid(sum)
}

func loadCSV(filename string) ([][]float64, []float64, map[string]int, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, nil, nil, err
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        return nil, nil, nil, err
    }

    vocabulary := make(map[string]int)
    vocabIndex := 0

    // First pass to build vocabulary
    for _, record := range records {
        for i, value := range record {
            if i < len(record)-1 {
                words := strings.Fields(value)
                for _, word := range words {
                    if _, exists := vocabulary[word]; !exists {
                        vocabulary[word] = vocabIndex
                        vocabIndex++
                    }
                }
            }
        }
    }

    var X [][]float64
    var y []float64

    // Second pass to build feature vectors
    for _, record := range records {
        features := make([]float64, len(vocabulary))
        for i, value := range record {
            if i == len(record)-1 {
                label, err := strconv.ParseFloat(value, 64)
                if err != nil {
                    return nil, nil, nil, err
                }
                y = append(y, label)
            } else {
                words := strings.Fields(value)
                for _, word := range words {
                    if index, exists := vocabulary[word]; exists {
                        features[index]++
                    }
                }
            }
        }
        X = append(X, features)
    }

    return X, y, vocabulary, nil
}

func main() {
    // Load data from CSV
    X, y, vocabulary, err := loadCSV("data.csv")
    if err != nil {
        log.Fatal(err)
    }

    // Create logistic regression model
    lr := NewLogisticRegression(len(vocabulary))

    // Channel to signal when training is done
    done := make(chan bool)

    // Train the model in a separate Go routine
    go lr.train(X, y, 0.1, 1000, done)

    // Wait for the model to be trained
    <-done

    // Read input from the console and make predictions
    reader := bufio.NewReader(os.Stdin)
    for {
        fmt.Print("Enter a sample text: ")
        sampleText, _ := reader.ReadString('\n')
        sampleText = strings.TrimSpace(sampleText)

        newSample := make([]float64, len(vocabulary))
        words := strings.Fields(sampleText)
        for _, word := range words {
            if index, exists := vocabulary[word]; exists {
                newSample[index]++
            }
        }

        prediction := lr.predict(newSample)
				if prediction < 0.3 { 
					fmt.Printf("FAIL '%s': %.4f\n", sampleText, prediction)
				} else { 
					fmt.Printf("PASS '%s': %.4f\n", sampleText, prediction)
				}
    }
}
