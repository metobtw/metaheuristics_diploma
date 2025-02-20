package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	gofourier "github.com/ardabasaran/go-fourier"
	"gocv.io/x/gocv"
)

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func sign(x float64) float64 {
	if x < 0 {
		return -1.0
	}
	return 1.0
}

func saveInfo(arr *[]int, path string) {
	slice := *arr
	strOut := ""
	for i := 0; i < len(slice); i++ {
		strOut += strconv.Itoa(slice[i])
		if i != len(slice)-1 {
			strOut += " "
		}
	}
	f, _ := os.Create(path)
	_, err := f.WriteString(strOut)
	if err != nil {
		return
	}
	err = f.Close()
	if err != nil {
		return
	}
}

func saveQ(q float64, path string) {
	f, _ := os.Create(path)
	_, err := f.WriteString(strconv.FormatFloat(q, 'f', -1, 64))
	if err != nil {
		return
	}
	err = f.Close()
	if err != nil {
		return
	}
}

func loadQ(path string) float64 {
	data, _ := os.ReadFile(path)
	qRet, _ := strconv.ParseFloat(string(data), 64)
	return qRet
}

func doDct(inputArr *[][]int) [][]float64 {
	slice := *inputArr

	floatInput := make([][]float64, len(slice))

	for i := 0; i < len(slice); i++ {
		floatInput[i] = make([]float64, len(slice[0]))
		for j := 0; j < len(slice[0]); j++ {
			floatInput[i][j] = float64(slice[i][j])
		}
	}

	dct2dReturn, _ := gofourier.DCT2D(floatInput)

	return dct2dReturn
}

func embedToDct(dctMatrix [][]float64, bitString string, mode string, q float64) [][]float64 {
	ind := 0
	cntj := 6
	for i := 0; i < len(dctMatrix); i++ {
		for j := len(dctMatrix[0]) - 1; j > cntj; j-- {
			value := dctMatrix[i][j]
			signValue := sign(value)
			magnitude := math.Abs(value) / q
			bit := int(bitString[ind] - '0')

			dctMatrix[i][j] = signValue * ((q * math.Floor(magnitude)) + (q/2)*float64(bit))

			if mode != "A" {
				return dctMatrix
			}
			ind++
		}
		if i == 3 {
			continue
		}
		cntj--
	}
	return dctMatrix
}

func undoDct(dct2din *[][]float64) [][]int {
	slice := *dct2din
	idctArr, _ := gofourier.DCTInverse2D(slice)

	outputResult := make([][]int, len(idctArr))
	for i := 0; i < len(idctArr); i++ {
		outputResult[i] = make([]int, len(idctArr[0]))
		for j := 0; j < len(idctArr[0]); j++ {
			outputResult[i][j] = int(idctArr[i][j])
		}
	}
	return outputResult
}

func generateQ(inputArr *[][]int) float64 {
	dctArr := doDct(inputArr)
	valMap := make(map[int]int)
	for i := 0; i < len(dctArr); i++ {
		for j := 0; j < len(dctArr[0]); j++ {
			rounded := int(math.Round(dctArr[i][j]))
			if rounded <= 20 && rounded >= 3 {
				valMap[rounded]++
			}
		}
	}
	retQ, cnt := 0, 65 // 8 * 8 block = 64
	for key, value := range valMap {
		if value <= cnt {
			cnt = value
			if key < retQ {
				retQ = key
			}
		}
	}
	return float64(retQ)
}

func generatePopulation(origMatrix *[][]int, embeddedMatrix *[][]int, populationSize int, beta float64, searchSpace int) [][]float64 {
	origSlice := *origMatrix
	embedSlice := *embeddedMatrix
	var ind = 0
	difference := make([]float64, len(origSlice)*len(origSlice[0]))
	for i := 0; i < len(origSlice); i++ {
		for j := 0; j < len(origSlice[0]); j++ {
			difference[ind] = float64(origSlice[i][j] - embedSlice[i][j]) // pochemu float
			ind++
		}
	}
	population := make([][]float64, populationSize)
	for i := 0; i < populationSize; i++ {
		population[i] = make([]float64, len(difference))
		for j := 0; j < len(difference); j++ {
			if i == 0 {
				population[i][j] = difference[j]
			} else {
				randomValue := rand.Float64()
				if randomValue > beta {
					randomSearch := rand.Intn(2*searchSpace+1) - searchSpace
					population[i][j] = float64(randomSearch)
				} else {
					population[i][j] = difference[j]
				}
			}
		}
	}
	return population
}

func extractingDct(pixelBlock *[][]int, q float64) string {
	dctBlock := doDct(pixelBlock)
	s := strings.Builder{}

	cntj := 6
	for i := 0; i < len(dctBlock); i++ {
		for j := len(dctBlock[0]) - 1; j > cntj; j-- {
			c0 := sign(dctBlock[i][j]) * (q*math.Floor(math.Abs(dctBlock[i][j])/q) + (q/2)*0)
			c1 := sign(dctBlock[i][j]) * (q*math.Floor(math.Abs(dctBlock[i][j])/q) + (q/2)*1)
			if math.Abs(dctBlock[i][j]-c0) < math.Abs(dctBlock[i][j]-c1) {
				s.WriteByte('0')
				if s.String() == "0" {
					return "0"
				}
			} else {
				s.WriteByte('1')
			}
		}
		if i == 3 {
			continue
		}
		cntj--
	}
	return s.String()
}

type ReturnMetric struct {
	computedMetric float64
	resultArray    []float64
}

type Metric struct {
	blockMatrix *[][]int
	bitString   string
	searchSpace int
	mode        string
	q           float64
}

func (m Metric) ComputeMetric(block *[]float64) ReturnMetric {
	sliceBlockMatrix := *m.blockMatrix
	sliceBlock := *block

	newBlock := make([][]int, len(sliceBlockMatrix))
	for i := range sliceBlockMatrix {
		newBlock[i] = make([]int, len(sliceBlockMatrix[i]))
		copy(newBlock[i], sliceBlockMatrix[i])
	}
	blockFlatten := make([]float64, len(sliceBlock))
	copy(blockFlatten, sliceBlock)

	for i := range blockFlatten {
		blockFlatten[i] = math.Floor(blockFlatten[i])
		if math.Abs(blockFlatten[i]) > float64(m.searchSpace) {
			blockFlatten[i] = float64(rand.Intn(2*m.searchSpace+1) - m.searchSpace)
		}
	}
	indFl := 0
	for i := 0; i < len(newBlock); i++ {
		for j := 0; j < len(newBlock); j++ {
			newBlock[i][j] -= int(blockFlatten[indFl])
			if newBlock[i][j] > 255 {
				diff := math.Abs(float64(newBlock[i][j] - 255))
				blockFlatten[indFl] += diff
				newBlock[i][j] = 255
			}
			if newBlock[i][j] < 0 {
				diff := math.Abs(float64(newBlock[i][j]))
				blockFlatten[indFl] -= diff
				newBlock[i][j] = 0
			}
			indFl++
		}
	}

	sumElem := 0.0
	for i := 0; i < len(newBlock); i++ {
		for j := 0; j < len(newBlock); j++ {
			sumElem += math.Pow(float64(sliceBlockMatrix[i][j])-float64(newBlock[i][j]), 2)
		}
	}
	var psnr float64
	if sumElem != 0 {
		psnr = 10 * math.Log10((math.Pow(8, 2)*math.Pow(255, 2))/sumElem)
	} else {
		psnr = 42
	}
	s := extractingDct(&newBlock, m.q)
	cnt := 0
	if s[0] == m.bitString[0] {
		for i := 0; i < len(s); i++ {
			if s[i] == m.bitString[i] {
				cnt++
			}
		}
	}
	return ReturnMetric{psnr/10000 + float64(cnt)/float64(len(s)), blockFlatten}
}

type Metaheuristic struct {
	populationSize int
	numIterations  int
	numFeatures    int
	metric         *Metric
}
type DE struct {
	agents            [][]float64
	metaheuristicInfo Metaheuristic
	cr                float64
	f                 float64
}

func (meta *DE) optimize() ReturnMetric {
	fitness := make([]float64, meta.metaheuristicInfo.populationSize)
	for i := 0; i < meta.metaheuristicInfo.populationSize; i++ {
		metricValue := meta.metaheuristicInfo.metric.ComputeMetric(&meta.agents[i])
		meta.agents[i] = metricValue.resultArray
		fitness[i] = metricValue.computedMetric
	}
	bestAgentFitness := 0.0
	bestAgent := make([]float64, len(meta.agents[0]))
	y := make([]float64, len(meta.agents[0]))
	for t := 0; t < meta.metaheuristicInfo.numIterations; t++ {
		for i := 0; i < len(meta.agents); i++ {
			var aInd, bInd, cInd int
			aInd = rand.Intn(len(meta.agents))
			bInd = rand.Intn(len(meta.agents))
			cInd = rand.Intn(len(meta.agents))
			for aInd == i {
				aInd = rand.Intn(len(meta.agents))
			}
			for bInd == i || bInd == aInd {
				bInd = rand.Intn(len(meta.agents))
			}
			for cInd == i || cInd == aInd || cInd == bInd {
				cInd = rand.Intn(len(meta.agents))
			}
			for pos := 0; pos < len(meta.agents[0]); pos++ {
				r := rand.Float64()
				if r < meta.cr {
					y[pos] = meta.agents[aInd][pos] + meta.f*(meta.agents[bInd][pos]-meta.agents[cInd][pos])
				} else {
					y[pos] = meta.agents[i][pos]
				}
			}
			metricValue := meta.metaheuristicInfo.metric.ComputeMetric(&y)
			if metricValue.computedMetric > fitness[i] {
				fitness[i] = metricValue.computedMetric
				meta.agents[i] = metricValue.resultArray
				if fitness[i] > bestAgentFitness {
					bestAgentFitness = fitness[i]
					bestAgent = meta.agents[i]
				}
			}
		}
	}
	return ReturnMetric{bestAgentFitness, bestAgent}
}

type SCA struct {
	agents            [][]float64
	metaheuristicInfo Metaheuristic
	aLinearComponent  float64
}

func (meta *SCA) optimize() ReturnMetric {
	fitness := make([]float64, meta.metaheuristicInfo.populationSize)
	for i := 0; i < meta.metaheuristicInfo.populationSize; i++ {
		metricValue := meta.metaheuristicInfo.metric.ComputeMetric(&meta.agents[i])
		meta.agents[i] = metricValue.resultArray
		fitness[i] = metricValue.computedMetric
	}

	bestAgentIndex := 0
	for i := 0; i < meta.metaheuristicInfo.populationSize; i++ {
		if fitness[i] > fitness[bestAgentIndex] {
			bestAgentIndex = i
		}
	}

	bestAgentFitness := fitness[bestAgentIndex]
	bestAgent := meta.agents[bestAgentIndex]
	for t := 0; t < meta.metaheuristicInfo.numIterations; t++ {
		for i := 0; i < len(meta.agents); i++ {
			aT := meta.aLinearComponent - float64(t)*(meta.aLinearComponent/float64(meta.metaheuristicInfo.numIterations))
			r1 := rand.Float64()
			r2 := rand.Float64()
			A := 2*aT*r1 - aT
			C := 2 * r2
			randomAgentIndex := rand.Intn(len(meta.agents))
			for randomAgentIndex == i {
				randomAgentIndex = rand.Intn(len(meta.agents))
			}
			randomAgent := meta.agents[randomAgentIndex]
			D := make([]float64, meta.metaheuristicInfo.numFeatures)
			for ind := 0; ind < meta.metaheuristicInfo.numFeatures; ind++ {
				D[ind] = math.Abs(C*randomAgent[ind] - meta.agents[i][ind])
			}
			newPosition := make([]float64, meta.metaheuristicInfo.numFeatures)
			for ind := 0; ind < meta.metaheuristicInfo.numFeatures; ind++ {
				newPosition[ind] = randomAgent[ind] - D[ind]*A
			}

			metricValue := meta.metaheuristicInfo.metric.ComputeMetric(&newPosition)
			if metricValue.computedMetric > fitness[i] {
				fitness[i] = metricValue.computedMetric
				meta.agents[i] = metricValue.resultArray
				if fitness[i] > bestAgentFitness {
					bestAgentFitness = fitness[i]
					bestAgent = meta.agents[i]
				}
			}
		}
	}
	return ReturnMetric{bestAgentFitness, bestAgent}
}

type WOA struct {
	agents            [][]float64
	metaheuristicInfo Metaheuristic
	searching         float64
}

func (meta *WOA) optimize() ReturnMetric {
	fitness := make([]float64, meta.metaheuristicInfo.populationSize)
	for i := 0; i < meta.metaheuristicInfo.populationSize; i++ {
		metricValue := meta.metaheuristicInfo.metric.ComputeMetric(&meta.agents[i])
		meta.agents[i] = metricValue.resultArray
		fitness[i] = metricValue.computedMetric
	}
	bestAgentFitness := 0.0
	bestAgent := make([]float64, len(meta.agents[0]))

	for t := 0; t < meta.metaheuristicInfo.numIterations; t++ {
		a := 2.0 - float64(t)*(2.0/float64(meta.metaheuristicInfo.numIterations))
		for i := 0; i < len(meta.agents); i++ {
			r1 := rand.Float64()
			r2 := rand.Float64()
			A := 2*a*r1 - a
			C := 2 * r2
			b := 1.0
			l := rand.Float64()*2 - 1
			p := rand.Float64()
			XRand := meta.agents[rand.Intn(len(meta.agents))]
			DXRand := make([]float64, meta.metaheuristicInfo.numFeatures)
			XNew := make([]float64, meta.metaheuristicInfo.numFeatures)

			if p < 0.5 {
				if math.Abs(A) < 1 {
					for j := 0; j < meta.metaheuristicInfo.numFeatures; j++ {
						DXRand[j] = math.Abs(C*XRand[j] - meta.agents[i][j])
						XNew[j] = XRand[j] - A*DXRand[j]
					}
				} else {
					for j := 0; j < meta.metaheuristicInfo.numFeatures; j++ {
						XNew[j] = XRand[j] - A*math.Abs(C*XRand[j]-meta.agents[i][j])
					}
				}
			} else {
				for j := 0; j < meta.metaheuristicInfo.numFeatures; j++ {
					DXRand[j] = math.Abs(XRand[j] - meta.agents[i][j])
					XNew[j] = DXRand[j]*math.Exp(b*l)*math.Cos(2*math.Pi*l) + XRand[j]
				}
			}

			for j := 0; j < meta.metaheuristicInfo.numFeatures; j++ {
				XNew[j] = clamp(XNew[j], -meta.searching, meta.searching)
			}

			metricValue := meta.metaheuristicInfo.metric.ComputeMetric(&XNew)

			if metricValue.computedMetric > fitness[i] {
				fitness[i] = metricValue.computedMetric
				meta.agents[i] = metricValue.resultArray
				if fitness[i] > bestAgentFitness {
					bestAgentFitness = fitness[i]
					bestAgent = meta.agents[i]
				}
			}
		}
	}
	return ReturnMetric{bestAgentFitness, bestAgent}
}

func psnr(originalImg *[][]int, savedImg *[][]int) float64 {
	sumElements := 0.0
	for i := 0; i < len(*originalImg); i++ {
		for j := 0; j < len((*originalImg)[i]); j++ {
			sumElements += math.Pow(float64((*originalImg)[i][j]-(*savedImg)[i][j]), 2)
		}
	}
	return 10 * math.Log10(math.Pow(255.0, 4)/sumElements)
}

func ssim(originalImg *[][]int, savedImg *[][]int) float64 {
	var mean1, mean2 = 0.0, 0.0
	for i := 0; i < len(*originalImg); i++ {
		for j := 0; j < len((*originalImg)[i]); j++ {
			mean1 += float64((*originalImg)[i][j])
			mean2 += float64((*savedImg)[i][j])
		}
	}
	mean1 /= float64(len(*originalImg) * len(*originalImg) * 3)
	mean2 /= float64(len(*originalImg) * len(*originalImg) * 3)
	var sd1, sd2, cov = 0.0, 0.0, 0.0
	for i := 0; i < len(*savedImg); i++ {
		for j := 0; j < len((*savedImg)[i]); j++ {
			sd1 += math.Pow(float64((*originalImg)[i][j])/3-mean1, 2)
			sd2 += math.Pow(float64((*savedImg)[i][j])/3-mean2, 2)
			cov += (float64((*originalImg)[i][j])/3 - mean1) * (float64((*savedImg)[i][j])/3 - mean2)
		}
	}
	cov /= float64(len(*savedImg) * len(*savedImg))
	sd1 = math.Pow(sd1/(float64(len(*savedImg)*len(*savedImg))), 0.5)
	sd2 = math.Pow(sd2/(float64(len(*savedImg)*len(*savedImg))), 0.5)
	c1 := math.Pow(0.01*255, 2)
	c2 := math.Pow(0.03*255, 2)
	return ((2*mean1*mean2 + c1) * (2*cov + c2)) / ((math.Pow(mean1, 2) + math.Pow(mean2, 2) + c1) * (math.Pow(sd1, 2) + math.Pow(sd2, 2) + c2))
}

var SearchSpace = 10

type MetaheuristicB struct {
	populationSize int
	numIterations  int
	numFeatures    int
	metric         *Metric
	agents         [][]float64
	optimizeFunc   func(*MetaheuristicB, ...interface{}) ReturnMetric
	params         interface{}
}

func optimizeWoa(meta *MetaheuristicB, args ...interface{}) ReturnMetric {
	searching, _ := args[0].(float64)
	fitness := make([]float64, meta.populationSize)
	for i := 0; i < meta.populationSize; i++ {
		metricValue := meta.metric.ComputeMetric(&meta.agents[i])
		meta.agents[i] = metricValue.resultArray
		fitness[i] = metricValue.computedMetric
	}
	bestAgentFitness := 0.0
	bestAgent := make([]float64, len(meta.agents[0]))

	for t := 0; t < meta.numIterations; t++ {
		a := 2.0 - float64(t)*(2.0/float64(meta.numIterations))
		for i := 0; i < len(meta.agents); i++ {
			r1 := rand.Float64()
			r2 := rand.Float64()
			A := 2*a*r1 - a
			C := 2 * r2
			b := 1.0
			l := rand.Float64()*2 - 1
			p := rand.Float64()
			XRand := meta.agents[rand.Intn(len(meta.agents))]
			DXRand := make([]float64, meta.numFeatures)
			XNew := make([]float64, meta.numFeatures)

			if p < 0.5 {
				if math.Abs(A) < 1 {
					for j := 0; j < meta.numFeatures; j++ {
						DXRand[j] = math.Abs(C*XRand[j] - meta.agents[i][j])
						XNew[j] = XRand[j] - A*DXRand[j]
					}
				} else {
					for j := 0; j < meta.numFeatures; j++ {
						XNew[j] = XRand[j] - A*math.Abs(C*XRand[j]-meta.agents[i][j])
					}
				}
			} else {
				for j := 0; j < meta.numFeatures; j++ {
					DXRand[j] = math.Abs(XRand[j] - meta.agents[i][j])
					XNew[j] = DXRand[j]*math.Exp(b*l)*math.Cos(2*math.Pi*l) + XRand[j]
				}
			}

			for j := 0; j < meta.numFeatures; j++ {
				XNew[j] = clamp(XNew[j], -searching, searching)
			}

			metricValue := meta.metric.ComputeMetric(&XNew)

			if metricValue.computedMetric > fitness[i] {
				fitness[i] = metricValue.computedMetric
				meta.agents[i] = metricValue.resultArray
				if fitness[i] > bestAgentFitness {
					bestAgentFitness = fitness[i]
					bestAgent = meta.agents[i]
				}
			}
		}
	}
	return ReturnMetric{bestAgentFitness, bestAgent}
}

func optimizeDe(meta *MetaheuristicB, args ...interface{}) ReturnMetric {
	cr := args[0].(float64)
	f := args[1].(float64)
	fitness := make([]float64, meta.populationSize)
	for i := 0; i < meta.populationSize; i++ {
		metricValue := meta.metric.ComputeMetric(&meta.agents[i])
		meta.agents[i] = metricValue.resultArray
		fitness[i] = metricValue.computedMetric
	}
	bestAgentFitness := 0.0
	bestAgent := make([]float64, len(meta.agents[0]))
	y := make([]float64, len(meta.agents[0]))
	for t := 0; t < meta.numIterations; t++ {
		for i := 0; i < len(meta.agents); i++ {
			var aInd, bInd, cInd int
			aInd = rand.Intn(len(meta.agents))
			bInd = rand.Intn(len(meta.agents))
			cInd = rand.Intn(len(meta.agents))
			for aInd == i {
				aInd = rand.Intn(len(meta.agents))
			}
			for bInd == i || bInd == aInd {
				bInd = rand.Intn(len(meta.agents))
			}
			for cInd == i || cInd == aInd || cInd == bInd {
				cInd = rand.Intn(len(meta.agents))
			}
			for pos := 0; pos < len(meta.agents[0]); pos++ {
				r := rand.Float64()
				if r < cr {
					y[pos] = meta.agents[aInd][pos] + f*(meta.agents[bInd][pos]-meta.agents[cInd][pos])
				} else {
					y[pos] = meta.agents[i][pos]
				}
			}
			metricValue := meta.metric.ComputeMetric(&y)
			if metricValue.computedMetric > fitness[i] {
				fitness[i] = metricValue.computedMetric
				meta.agents[i] = metricValue.resultArray
				if fitness[i] > bestAgentFitness {
					bestAgentFitness = fitness[i]
					bestAgent = meta.agents[i]
				}
			}
		}
	}
	return ReturnMetric{bestAgentFitness, bestAgent}
}

func optimizeSca(meta *MetaheuristicB, args ...interface{}) ReturnMetric {
	aLinearComponent := args[0].(float64)
	fitness := make([]float64, meta.populationSize)
	for i := 0; i < meta.populationSize; i++ {
		metricValue := meta.metric.ComputeMetric(&meta.agents[i])
		meta.agents[i] = metricValue.resultArray
		fitness[i] = metricValue.computedMetric
	}

	bestAgentIndex := 0
	for i := 0; i < meta.populationSize; i++ {
		if fitness[i] > fitness[bestAgentIndex] {
			bestAgentIndex = i
		}
	}

	bestAgentFitness := fitness[bestAgentIndex]
	bestAgent := meta.agents[bestAgentIndex]
	for t := 0; t < meta.numIterations; t++ {
		for i := 0; i < len(meta.agents); i++ {
			aT := aLinearComponent - float64(t)*(aLinearComponent/float64(meta.numIterations))
			r1 := rand.Float64()
			r2 := rand.Float64()
			A := 2*aT*r1 - aT
			C := 2 * r2
			randomAgentIndex := rand.Intn(len(meta.agents))
			for randomAgentIndex == i {
				randomAgentIndex = rand.Intn(len(meta.agents))
			}
			randomAgent := meta.agents[randomAgentIndex]
			D := make([]float64, meta.numFeatures)
			for ind := 0; ind < meta.numFeatures; ind++ {
				D[ind] = math.Abs(C*randomAgent[ind] - meta.agents[i][ind])
			}
			newPosition := make([]float64, meta.numFeatures)
			for ind := 0; ind < meta.numFeatures; ind++ {
				newPosition[ind] = randomAgent[ind] - D[ind]*A
			}

			metricValue := meta.metric.ComputeMetric(&newPosition)
			if metricValue.computedMetric > fitness[i] {
				fitness[i] = metricValue.computedMetric
				meta.agents[i] = metricValue.resultArray
				if fitness[i] > bestAgentFitness {
					bestAgentFitness = fitness[i]
					bestAgent = meta.agents[i]
				}
			}
		}
	}
	return ReturnMetric{bestAgentFitness, bestAgent}
}

// var q float64 = 3 + rand.Float64()*(20-3)

var PopulationSize = 128
var NumIterations = 128
var NumFeatures = 64

func main() {

	scanner := bufio.NewScanner(os.Stdin)

	//pictures := [...]string{"peppers512.png", "lena512.png", "airplane512.png", "baboon512.png", "barbara512.png", "boat512.png", "goldhill512.png", "stream_and_bridge512.png"}
	pictures := [...]string{"peppers512.png"}

	//metaheuristics := [...]string{"de", "sca", "woa"}
	metaheuristics := [...]string{"woa"}

	scanner.Scan()
	mode := scanner.Text()

	for _, picture := range pictures {
		if mode == "metrics" {
			fmt.Println(picture)
		}
		for _, metaheuristic := range metaheuristics {
			dirPath := metaheuristic + "_" + picture
			err := os.Mkdir(dirPath, 0777)
			if err != nil {
				return
			}

			if mode == "embedding" {
				q := 3 + rand.Float64()*(20-3)
				fileContent, _ := os.ReadFile("to_embed.txt")
				information := string(fileContent)
				indInformation := 0
				var image = gocv.IMRead(picture, gocv.IMReadGrayScale)
				var rows, cols = image.Rows(), image.Cols()
				matData, _ := image.DataPtrUint8()
				fmt.Print(rows, cols)
				img := make([][]int, rows)
				for row := range img {
					img[row] = make([]int, cols)
					for col := range img[row] {
						img[row][col] = int(matData[row*cols+col])
					}
				}
				// нужны ли вообще блоки?
				blocks := rand.Perm(rows * cols / 64)
				saveInfo(&blocks, dirPath+"/blocks.txt")

				copyImg := make([][]int, len(img))
				for i := range img {
					copyImg[i] = make([]int, len(img[i]))
					copy(copyImg[i], img[i])
				}
				saveQ(q, dirPath+"/q.txt")
				var cnt1, cntBlocks int
				for _, block := range blocks {
					fmt.Print("\n", cntBlocks, "\n")
					cntBlocks++
					pixelMatrix := make([][]int, 8)
					blockW := block % (rows / 8)
					blockH := (block - blockW) / (rows / 8)
					for i1 := blockH * 8; i1 < blockH*8+8; i1++ {
						pixelMatrix[i1%8] = make([]int, 8)
						for i2 := blockW * 8; i2 < blockW*8+8; i2++ {
							pixelMatrix[i1%8][i2%8] = img[i1][i2]
						}
					}
					dctMatrix := doDct(&pixelMatrix)
					dctMatrixNew := embedToDct(dctMatrix, "1"+information[indInformation:indInformation+31], "A", q)
					newPixelMatrix := undoDct(&dctMatrixNew)
					metric := Metric{&pixelMatrix, "1" + information[indInformation:indInformation+31], SearchSpace, "A", q}
					metaheuristicInfo := Metaheuristic{PopulationSize, NumIterations, NumFeatures, &metric}
					var solution ReturnMetric
					if metaheuristic == "de" {
						population := generatePopulation(&pixelMatrix, &newPixelMatrix, 128, 0.9, SearchSpace)
						//de := DE{population, metaheuristic_info, 0.3, 0.1}
						de := MetaheuristicB{PopulationSize, NumIterations, NumFeatures, &metric, population, optimizeDe, []interface{}{0.3, 0.1}}
						solution = de.optimizeFunc(&de, de.params)
					} else if metaheuristic == "sca" {
						population := generatePopulation(&pixelMatrix, &newPixelMatrix, 128, 0.9, SearchSpace)
						//de := SCA{population, metaheuristic_info, 2.0}
						de := MetaheuristicB{PopulationSize, NumIterations, NumFeatures, &metric, population, optimizeSca, []interface{}{2.0}}
						solution = de.optimizeFunc(&de, de.params)
						//solution = de.optimize()
					} else if metaheuristic == "woa" {
						population := generatePopulation(&pixelMatrix, &newPixelMatrix, 128, 0.9, SearchSpace)
						//de := WOA{population, metaheuristic_info, float64(SEARCH_SPACE)}
						de := MetaheuristicB{PopulationSize, NumIterations, NumFeatures, &metric, population, optimizeWoa, []interface{}{SearchSpace}}
						solution = de.optimizeFunc(&de, de.params)
					}
					if solution.computedMetric > 1 {
						cnt1++
						ind := 0
						for i1 := blockH * 8; i1 < blockH*8+8; i1++ {
							for i2 := blockW * 8; i2 < blockW*8+8; i2++ {
								copyImg[i1][i2] -= int(solution.resultArray[ind])
								ind++
							}
						}
						indInformation += 31
					} else {
						fmt.Print(solution.computedMetric)
						searching := 5
						dctMatrix = doDct(&pixelMatrix)
						dctMatrixNew = embedToDct(dctMatrix, "0", "Z", q)
						newPixelMatrix = undoDct(&dctMatrixNew)
						population := generatePopulation(&pixelMatrix, &newPixelMatrix, 128, 0.9, searching)
						metric = Metric{&pixelMatrix, "0", searching, "Z", q}
						metaheuristicInfo = Metaheuristic{PopulationSize, NumIterations, NumFeatures, &metric}
						de := DE{population, metaheuristicInfo, 0.3, 0.1}
						//de := SCA{population, metaheuristic_info, 2.0}
						//de := WOA{population, metaheuristic_info, float64(SEARCH_SPACE)}

						solution = de.optimize()
						for i1 := 0; i1 < 64; i1++ {
							fmt.Println(solution.resultArray[i1])
						}
						ind := 0
						for i1 := blockH * 8; i1 < blockH*8+8; i1++ {
							for i2 := blockW * 8; i2 < blockW*8+8; i2++ {
								copyImg[i1][i2] -= int(solution.resultArray[ind])
								ind++
							}
						}
					}
					q = generateQ(&copyImg)
				}
				imageMat := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8UC1)
				for row := 0; row < rows; row++ {
					for col := 0; col < cols; col++ {
						imageMat.SetUCharAt(row, col, uint8(copyImg[row][col]))
					}
				}
				outputFilePath := dirPath + "/saved.png"
				if ok := gocv.IMWrite(outputFilePath, imageMat); !ok {
					fmt.Println("Ошибка: не удалось сохранить изображение.")
					os.Exit(1)
				}
				err = imageMat.Close()
				if err != nil {
					return
				}
				fmt.Print(cnt1)
			} else if mode == "extraction" {
				q := loadQ(dirPath + "/q.txt")
				var bitString string
				var image = gocv.IMRead(dirPath+"/saved.png", gocv.IMReadGrayScale)
				var rows, cols = image.Rows(), image.Cols()
				matData, _ := image.DataPtrUint8()
				img := make([][]int, rows)
				for row := range img {
					img[row] = make([]int, cols)
					for col := range img[row] {
						img[row][col] = int(matData[row*cols+col])
					}
				}
				file, _ := os.Open(dirPath + "/blocks.txt")
				var blocks []int
				fileScanner := bufio.NewScanner(file)
				for fileScanner.Scan() {
					line := fileScanner.Text()
					for _, numStr := range strings.Split(line, " ") {
						num, _ := strconv.Atoi(numStr)
						blocks = append(blocks, num)
					}
				}
				for _, block := range blocks {
					pixelMatrix := make([][]int, 8)
					blockW := block % (rows / 8)
					blockH := (block - blockW) / (rows / 8)
					for i1 := blockH * 8; i1 < blockH*8+8; i1++ {
						pixelMatrix[i1%8] = make([]int, 8)
						for i2 := blockW * 8; i2 < blockW*8+8; i2++ {
							pixelMatrix[i1%8][i2%8] = img[i1][i2]
						}
					}
					s := extractingDct(&pixelMatrix, q)
					if s != "0" {
						bitString += s[1:]
					}
					q = generateQ(&pixelMatrix)
				}
				f, _ := os.Create(dirPath + "/saved.txt")
				_, err := f.WriteString(bitString)
				if err != nil {
					return
				}
				err = f.Close()
				if err != nil {
					return
				}
				err = file.Close()
				if err != nil {
					return
				}
			} else if mode == "metrics" {
				var image = gocv.IMRead(dirPath+"/saved.png", gocv.IMReadGrayScale)
				var rows, cols = image.Rows(), image.Cols()
				matData, _ := image.DataPtrUint8()
				img := make([][]int, rows)
				for row := range img {
					img[row] = make([]int, cols)
					for col := range img[row] {
						img[row][col] = int(matData[row*cols+col])
					}
				}
				image = gocv.IMRead(picture, gocv.IMReadGrayScale)
				rows, cols = image.Rows(), image.Cols()
				matData, _ = image.DataPtrUint8()
				imgBase := make([][]int, rows)
				for row := range imgBase {
					imgBase[row] = make([]int, cols)
					for col := range imgBase[row] {
						imgBase[row][col] = int(matData[row*cols+col])
					}
				}
				fileContent, _ := os.ReadFile(dirPath + "/saved.txt")
				information := string(fileContent)
				fmt.Println(metaheuristic, len(information), ssim(&imgBase, &img), psnr(&imgBase, &img))
			}
		}
	}
}
