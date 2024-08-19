package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

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
	X, y, vocabulary, err := loadCSV("data/data.csv")
	if err != nil {
		log.Fatal(err)
	}

	lr := NewLogisticRegression(len(vocabulary))

	done := make(chan bool)

	go lr.train(X, y, 0.1, 1000, done)

	<-done

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Enter a sample text: ")
		sampleText, _ := reader.ReadString('\n')
		sampleText = strings.TrimSpace(sampleText)

		words := strings.Fields(sampleText)
		for _, word := range words {
			newSample := make([]float64, len(vocabulary))
			if index, exists := vocabulary[strings.ToLower(word)]; exists {
				newSample[index]++
			}

			prediction := lr.predict(newSample)
			if prediction <= 0.3 {
				fmt.Printf("FAIL '%s': %.4f\n", word, prediction)
			} else {
				fmt.Printf("PASS '%s': %.4f\n", word, prediction)
			}
		}
	}
}
