package main

import (
	"math"
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

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func computeLoss(yTrue, yPred float64) float64 {
	if yTrue == 1 {
		return -math.Log(yPred)
	}
	return -math.Log(1 - yPred)
}
