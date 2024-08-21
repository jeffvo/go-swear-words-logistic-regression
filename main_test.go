package main

import (
	"log"
	"strings"
	"testing"
)

func TestPredict(t *testing.T) {
	// arrange
	X, y, vocabulary, err := loadCSV("data/test_cases.csv")
	if err != nil {
		log.Fatal(err)
	}

	lr := NewLogisticRegression(len(vocabulary))

	done := make(chan bool)

	//act
	go lr.train(X, y, 0.1, 1000, done)

	<-done

	// assert
	testCases := []struct {
		word     string
		expected string
	}{
		{"hello", "FAIL"},
		{"world", "PASS"},
		{"goodbye", "PASS"},
		{"unknown", "PASS"},
	}

	for _, testCase := range testCases {
		newSample := make([]float64, len(vocabulary))
		if index, exists := vocabulary[strings.ToLower(testCase.word)]; exists {
			newSample[index]++
		}

		prediction := lr.predict(newSample)
		result := "FAIL"
		if prediction >= 0.3 {
			result = "PASS"
		}

		if result != testCase.expected {
			t.Errorf("For word '%s', expected %s but got %s score %f", testCase.word, testCase.expected, result, prediction)
		}
	}
}
