package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	go_fourier "github.com/ardabasaran/go-fourier"
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

func save_info(arr *[]int, path string) {
	slice := *arr
	str_out := ""
	for i := 0; i < len(slice); i++ {
		str_out += strconv.Itoa(slice[i])
		if i != len(slice)-1 {
			str_out += " "
		}
	}
	f, _ := os.Create(path)
	f.WriteString(str_out)
	f.Close()
}

func save_q(q float64, path string) {
	f, _ := os.Create(path)
	f.WriteString(strconv.FormatFloat(q, 'f', -1, 64))
	f.Close()
}

func load_q(path string) float64 {
	data, _ := os.ReadFile(path)
	q_ret, _ := strconv.ParseFloat(string(data), 64)
	return q_ret
}

func do_dct(input_arr *[][]int) [][]float64 {
	slice := *input_arr

	float_input := make([][]float64, len(slice))

	for i := 0; i < len(slice); i++ {
		float_input[i] = make([]float64, len(slice[0]))
		for j := 0; j < len(slice[0]); j++ {
			float_input[i][j] = float64(slice[i][j])
		}
	}

	dct2d_return, _ := go_fourier.DCT2D(float_input)

	return dct2d_return
}

func embed_to_dct(dct_matrix [][]float64, bit_string string, mode string, q float64) [][]float64 {
	ind := 0
	cntj := 6
	for i := 0; i < len(dct_matrix); i++ {
		for j := len(dct_matrix[0]) - 1; j > cntj; j-- {
			value := dct_matrix[i][j]
			signValue := sign(value)
			magnitude := math.Abs(value) / q
			bit := int(bit_string[ind] - '0')

			dct_matrix[i][j] = signValue * ((q * math.Floor(magnitude)) + (q/2)*float64(bit))

			if mode != "A" {
				return dct_matrix
			}
			ind++
		}
		if i == 3 {
			continue
		}
		cntj--
	}
	return dct_matrix
}

func undo_dct(dct_2din *[][]float64) [][]int {
	slice := *dct_2din
	idct_arr, _ := go_fourier.DCTInverse2D(slice)

	output_result := make([][]int, len(idct_arr))
	for i := 0; i < len(idct_arr); i++ {
		output_result[i] = make([]int, len(idct_arr[0]))
		for j := 0; j < len(idct_arr[0]); j++ {
			output_result[i][j] = int(idct_arr[i][j])
		}
	}
	return output_result
}

func generate_q(input_arr *[][]int) float64 {
	dct_arr := do_dct(input_arr)
	val_map := make(map[int]int)
	for i := 0; i < len(dct_arr); i++ {
		for j := 0; j < len(dct_arr[0]); j++ {
			rounded := int(math.Round(dct_arr[i][j]))
			if rounded <= 20 && rounded >= 3 {
				val_map[rounded]++
			}
		}
	}
	ret_q, cnt := 0, 65 // 8 * 8 block = 64
	for key, value := range val_map {
		if value <= cnt {
			cnt = value
			if key < ret_q {
				ret_q = key
			}
		}
	}
	return float64(ret_q)
}

func generate_population(orig_matrix *[][]int, embedded_matrix *[][]int, population_size int, beta float64, search_space int) [][]float64 {
	orig_slice := *orig_matrix
	embed_slice := *embedded_matrix
	var ind = 0
	difference := make([]float64, len(orig_slice)*len(orig_slice[0]))
	for i := 0; i < len(orig_slice); i++ {
		for j := 0; j < len(orig_slice[0]); j++ {
			difference[ind] = float64(orig_slice[i][j] - embed_slice[i][j]) // pochemu float
			ind++
		}
	}
	population := make([][]float64, population_size)
	for i := 0; i < population_size; i++ {
		population[i] = make([]float64, len(difference))
		for j := 0; j < len(difference); j++ {
			if i == 0 {
				population[i][j] = difference[j]
			} else {
				random_value := rand.Float64()
				if random_value > beta {
					random_search := rand.Intn(2*search_space+1) - search_space
					population[i][j] = float64(random_search)
				} else {
					population[i][j] = difference[j]
				}
			}
		}
	}
	return population
}

func extracting_dct(pixel_block *[][]int, q float64) string {
	dct_block := do_dct(pixel_block)
	s := strings.Builder{}

	cntj := 6
	for i := 0; i < len(dct_block); i++ {
		for j := len(dct_block[0]) - 1; j > cntj; j-- {
			c0 := sign(dct_block[i][j]) * (q*float64(math.Floor(math.Abs(dct_block[i][j])/q)) + (q/2)*0)
			c1 := sign(dct_block[i][j]) * (q*float64(math.Floor(math.Abs(dct_block[i][j])/q)) + (q/2)*1)
			if math.Abs(dct_block[i][j]-c0) < math.Abs(dct_block[i][j]-c1) {
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

type Return_metric struct {
	computed_metric float64
	result_array    []float64
}

type Metric struct {
	block_matrix *[][]int
	bit_string   string
	search_space int
	mode         string
	q            float64
}

func (m Metric) Compute_metric(block *[]float64) Return_metric {
	slice_block_matrix := *m.block_matrix
	slice_block := *block

	new_block := make([][]int, len(slice_block_matrix))
	for i := range slice_block_matrix {
		new_block[i] = make([]int, len(slice_block_matrix[i]))
		copy(new_block[i], slice_block_matrix[i])
	}
	block_flatten := make([]float64, len(slice_block))
	copy(block_flatten, slice_block)

	for i := range block_flatten {
		block_flatten[i] = math.Floor(block_flatten[i])
		if math.Abs(block_flatten[i]) > float64(m.search_space) {
			block_flatten[i] = float64(rand.Intn(2*m.search_space+1) - m.search_space)
		}
	}
	ind_fl := 0
	for i := 0; i < len(new_block); i++ {
		for j := 0; j < len(new_block); j++ {
			new_block[i][j] -= int(block_flatten[ind_fl])
			if new_block[i][j] > 255 {
				diff := math.Abs(float64(new_block[i][j] - 255))
				block_flatten[ind_fl] += diff
				new_block[i][j] = 255
			}
			if new_block[i][j] < 0 {
				diff := math.Abs(float64(new_block[i][j]))
				block_flatten[ind_fl] -= diff
				new_block[i][j] = 0
			}
			ind_fl++
		}
	}

	sum_elem := 0.0
	for i := 0; i < len(new_block); i++ {
		for j := 0; j < len(new_block); j++ {
			sum_elem += math.Pow(float64(slice_block_matrix[i][j])-float64(new_block[i][j]), 2)
		}
	}
	var psnr float64
	if sum_elem != 0 {
		psnr = 10 * math.Log10((math.Pow(8, 2)*math.Pow(255, 2))/sum_elem)
	} else {
		psnr = 42
	}
	s := extracting_dct(&new_block, m.q)
	cnt := 0
	if s[0] == m.bit_string[0] {
		for i := 0; i < len(s); i++ {
			if s[i] == m.bit_string[i] {
				cnt++
			}
		}
	}
	return Return_metric{psnr/10000 + float64(cnt)/float64(len(s)), block_flatten}
}

type Metaheuristic struct {
	population_size int
	num_iterations  int
	num_features    int
	metric          *Metric
}
type DE struct {
	agents             [][]float64
	metaheuristic_info Metaheuristic
	cr                 float64
	f                  float64
}

func (meta *DE) optimize() Return_metric {
	fitness := make([]float64, meta.metaheuristic_info.population_size)
	for i := 0; i < meta.metaheuristic_info.population_size; i++ {
		metric_value := meta.metaheuristic_info.metric.Compute_metric(&meta.agents[i])
		meta.agents[i] = metric_value.result_array
		fitness[i] = metric_value.computed_metric
	}
	best_agent_fitness := 0.0
	best_agent := make([]float64, len(meta.agents[0]))
	y := make([]float64, len(meta.agents[0]))
	for t := 0; t < meta.metaheuristic_info.num_iterations; t++ {
		for i := 0; i < len(meta.agents); i++ {
			var a_ind, b_ind, c_ind int
			a_ind = rand.Intn(len(meta.agents))
			b_ind = rand.Intn(len(meta.agents))
			c_ind = rand.Intn(len(meta.agents))
			for a_ind == i {
				a_ind = rand.Intn(len(meta.agents))
			}
			for b_ind == i || b_ind == a_ind {
				b_ind = rand.Intn(len(meta.agents))
			}
			for c_ind == i || c_ind == a_ind || c_ind == b_ind {
				c_ind = rand.Intn(len(meta.agents))
			}
			for pos := 0; pos < len(meta.agents[0]); pos++ {
				r := rand.Float64()
				if r < meta.cr {
					y[pos] = meta.agents[a_ind][pos] + meta.f*(meta.agents[b_ind][pos]-meta.agents[c_ind][pos])
				} else {
					y[pos] = meta.agents[i][pos]
				}
			}
			metric_value := meta.metaheuristic_info.metric.Compute_metric(&y)
			if metric_value.computed_metric > fitness[i] {
				fitness[i] = metric_value.computed_metric
				meta.agents[i] = metric_value.result_array
				if fitness[i] > best_agent_fitness {
					best_agent_fitness = fitness[i]
					best_agent = meta.agents[i]
				}
			}
		}
	}
	return Return_metric{best_agent_fitness, best_agent}
}

type SCA struct {
	agents             [][]float64
	metaheuristic_info Metaheuristic
	a_linear_component float64
}

func (meta *SCA) optimize() Return_metric {
	fitness := make([]float64, meta.metaheuristic_info.population_size)
	for i := 0; i < meta.metaheuristic_info.population_size; i++ {
		metric_value := meta.metaheuristic_info.metric.Compute_metric(&meta.agents[i])
		meta.agents[i] = metric_value.result_array
		fitness[i] = metric_value.computed_metric
	}

	best_agent_index := 0
	for i := 0; i < meta.metaheuristic_info.population_size; i++ {
		if fitness[i] > fitness[best_agent_index] {
			best_agent_index = i
		}
	}

	best_agent_fitness := fitness[best_agent_index]
	best_agent := meta.agents[best_agent_index]
	for t := 0; t < meta.metaheuristic_info.num_iterations; t++ {
		for i := 0; i < len(meta.agents); i++ {
			a_t := meta.a_linear_component - float64(t)*(meta.a_linear_component/float64(meta.metaheuristic_info.num_iterations))
			r1 := rand.Float64()
			r2 := rand.Float64()
			A := 2*a_t*r1 - a_t
			C := 2 * r2
			random_agent_index := rand.Intn(len(meta.agents))
			for random_agent_index == i {
				random_agent_index = rand.Intn(len(meta.agents))
			}
			random_agent := meta.agents[random_agent_index]
			D := make([]float64, meta.metaheuristic_info.num_features)
			for ind := 0; ind < meta.metaheuristic_info.num_features; ind++ {
				D[ind] = math.Abs(C*random_agent[ind] - meta.agents[i][ind])
			}
			new_position := make([]float64, meta.metaheuristic_info.num_features)
			for ind := 0; ind < meta.metaheuristic_info.num_features; ind++ {
				new_position[ind] = random_agent[ind] - D[ind]*A
			}

			metric_value := meta.metaheuristic_info.metric.Compute_metric(&new_position)
			if metric_value.computed_metric > fitness[i] {
				fitness[i] = metric_value.computed_metric
				meta.agents[i] = metric_value.result_array
				if fitness[i] > best_agent_fitness {
					best_agent_fitness = fitness[i]
					best_agent = meta.agents[i]
				}
			}
		}
	}
	return Return_metric{best_agent_fitness, best_agent}
}

type WOA struct {
	agents             [][]float64
	metaheuristic_info Metaheuristic
	searching          float64
}

func (meta *WOA) optimize() Return_metric {
	fitness := make([]float64, meta.metaheuristic_info.population_size)
	for i := 0; i < meta.metaheuristic_info.population_size; i++ {
		metric_value := meta.metaheuristic_info.metric.Compute_metric(&meta.agents[i])
		meta.agents[i] = metric_value.result_array
		fitness[i] = metric_value.computed_metric
	}
	best_agent_fitness := 0.0
	best_agent := make([]float64, len(meta.agents[0]))

	for t := 0; t < meta.metaheuristic_info.num_iterations; t++ {
		a := 2.0 - float64(t)*(2.0/float64(meta.metaheuristic_info.num_iterations))
		for i := 0; i < len(meta.agents); i++ {
			r1 := rand.Float64()
			r2 := rand.Float64()
			A := 2*a*r1 - a
			C := 2 * r2
			b := 1.0
			l := rand.Float64()*2 - 1
			p := rand.Float64()
			X_rand := meta.agents[rand.Intn(len(meta.agents))]
			D_X_rand := make([]float64, meta.metaheuristic_info.num_features)
			X_new := make([]float64, meta.metaheuristic_info.num_features)

			if p < 0.5 {
				if math.Abs(A) < 1 {
					for j := 0; j < meta.metaheuristic_info.num_features; j++ {
						D_X_rand[j] = math.Abs(C*X_rand[j] - meta.agents[i][j])
						X_new[j] = X_rand[j] - A*D_X_rand[j]
					}
				} else {
					for j := 0; j < meta.metaheuristic_info.num_features; j++ {
						X_new[j] = X_rand[j] - A*math.Abs(C*X_rand[j]-meta.agents[i][j])
					}
				}
			} else {
				for j := 0; j < meta.metaheuristic_info.num_features; j++ {
					D_X_rand[j] = math.Abs(X_rand[j] - meta.agents[i][j])
					X_new[j] = D_X_rand[j]*math.Exp(b*l)*math.Cos(2*math.Pi*l) + X_rand[j]
				}
			}

			for j := 0; j < meta.metaheuristic_info.num_features; j++ {
				X_new[j] = clamp(X_new[j], -meta.searching, meta.searching)
			}

			metric_value := meta.metaheuristic_info.metric.Compute_metric(&X_new)

			if metric_value.computed_metric > fitness[i] {
				fitness[i] = metric_value.computed_metric
				meta.agents[i] = metric_value.result_array
				if fitness[i] > best_agent_fitness {
					best_agent_fitness = fitness[i]
					best_agent = meta.agents[i]
				}
			}
		}
	}
	return Return_metric{best_agent_fitness, best_agent}
}

func psnr(original_img *[][]int, saved_img *[][]int) float64 {
	sum_elements := 0.0
	for i := 0; i < len(*original_img); i++ {
		for j := 0; j < len((*original_img)[i]); j++ {
			sum_elements += math.Pow(float64((*original_img)[i][j]-(*saved_img)[i][j]), 2)
		}
	}
	return 10 * math.Log10(math.Pow(255.0, 4)/sum_elements)
}

func ssim(original_img *[][]int, saved_img *[][]int) float64 {
	var mean1, mean2 float64 = 0.0, 0.0
	for i := 0; i < len(*original_img); i++ {
		for j := 0; j < len((*original_img)[i]); j++ {
			mean1 += float64((*original_img)[i][j])
			mean2 += float64((*saved_img)[i][j])
		}
	}
	mean1 /= float64(len(*original_img) * len(*original_img) * 3)
	mean2 /= float64(len(*original_img) * len(*original_img) * 3)
	var sd1, sd2, cov float64 = 0.0, 0.0, 0.0
	for i := 0; i < len(*saved_img); i++ {
		for j := 0; j < len((*saved_img)[i]); j++ {
			sd1 += math.Pow(float64((*original_img)[i][j])/3-mean1, 2)
			sd2 += math.Pow(float64((*saved_img)[i][j])/3-mean2, 2)
			cov += (float64((*original_img)[i][j])/3 - mean1) * (float64((*saved_img)[i][j])/3 - mean2)
		}
	}
	cov /= float64(len(*saved_img) * len(*saved_img))
	sd1 = math.Pow(sd1/(float64(len(*saved_img)*len(*saved_img))), 0.5)
	sd2 = math.Pow(sd2/(float64(len(*saved_img)*len(*saved_img))), 0.5)
	c1 := math.Pow(0.01*255, 2)
	c2 := math.Pow(0.03*255, 2)
	return ((2*mean1*mean2 + c1) * (2*cov + c2)) / ((math.Pow(mean1, 2) + math.Pow(mean2, 2) + c1) * (math.Pow(sd1, 2) + math.Pow(sd2, 2) + c2))
}

var SEARCH_SPACE int = 10

// var q float64 = 3 + rand.Float64()*(20-3)
var POPULATION_SIZE int = 128
var NUM_ITERATIONS int = 128
var NUM_FEATURES int = 64

func main() {

	scanner := bufio.NewScanner(os.Stdin)

	//pictures := [...]string{"peppers512.png", "lena512.png", "airplane512.png", "baboon512.png", "barbara512.png", "boat512.png", "goldhill512.png", "stream_and_bridge512.png"}
	pictures := [...]string{"peppers512.png"}

	metaheuristics := [...]string{"de", "sca", "woa"}
	scanner.Scan()
	mode := scanner.Text()

	for _, picture := range pictures {
		if mode == "metrics" {
			fmt.Println(picture)
		}
		for _, metaheuristic := range metaheuristics {
			dir_path := metaheuristic + "_" + picture
			os.Mkdir(dir_path, 0777)

			if mode == "embedding" {
				q := 3 + rand.Float64()*(20-3)
				file_content, _ := os.ReadFile("to_embed.txt")
				information := string(file_content)
				ind_information := 0
				var image gocv.Mat = gocv.IMRead(picture, gocv.IMReadGrayScale)
				var rows, cols int = image.Rows(), image.Cols()
				mat_data, _ := image.DataPtrUint8()
				fmt.Print(rows, cols)
				img := make([][]int, rows)
				for row := range img {
					img[row] = make([]int, cols)
					for col := range img[row] {
						img[row][col] = int(mat_data[row*cols+col])
					}
				}
				// нужны ли вообще блоки?
				blocks := rand.Perm(rows * cols / 64)
				save_info(&blocks, dir_path+"/blocks.txt")

				copy_img := make([][]int, len(img))
				for i := range img {
					copy_img[i] = make([]int, len(img[i]))
					copy(copy_img[i], img[i])
				}
				save_q(q, dir_path+"/q.txt")
				var cnt1, cnt_blocks int
				for _, block := range blocks {
					fmt.Print("\n", cnt_blocks, "\n")
					cnt_blocks++
					pixel_matrix := make([][]int, 8)
					block_w := block % (rows / 8)
					block_h := (block - block_w) / (rows / 8)
					for i1 := block_h * 8; i1 < block_h*8+8; i1++ {
						pixel_matrix[i1%8] = make([]int, 8)
						for i2 := block_w * 8; i2 < block_w*8+8; i2++ {
							pixel_matrix[i1%8][i2%8] = img[i1][i2]
						}
					}
					dct_matrix := do_dct(&pixel_matrix)
					dct_matrix_new := embed_to_dct(dct_matrix, "1"+information[ind_information:ind_information+31], "A", q)
					new_pixel_matrix := undo_dct(&dct_matrix_new)
					metric := Metric{&pixel_matrix, "1" + information[ind_information:ind_information+31], SEARCH_SPACE, "A", q}
					metaheuristic_info := Metaheuristic{POPULATION_SIZE, NUM_ITERATIONS, NUM_FEATURES, &metric}
					var solution Return_metric
					if metaheuristic == "de" {
						population := generate_population(&pixel_matrix, &new_pixel_matrix, 128, 0.9, SEARCH_SPACE)
						de := DE{population, metaheuristic_info, 0.3, 0.1}
						solution = de.optimize()
					} else if metaheuristic == "sca" {
						population := generate_population(&pixel_matrix, &new_pixel_matrix, 128, 0.9, SEARCH_SPACE)
						de := SCA{population, metaheuristic_info, 2.0}
						solution = de.optimize()
					} else if metaheuristic == "woa" {
						population := generate_population(&pixel_matrix, &new_pixel_matrix, 128, 0.9, SEARCH_SPACE)
						de := WOA{population, metaheuristic_info, float64(SEARCH_SPACE)}
						solution = de.optimize()
					}
					if solution.computed_metric > 1 {
						cnt1++
						ind := 0
						for i1 := block_h * 8; i1 < block_h*8+8; i1++ {
							for i2 := block_w * 8; i2 < block_w*8+8; i2++ {
								copy_img[i1][i2] -= int(solution.result_array[ind])
								ind++
							}
						}
						ind_information += 31
					} else {
						fmt.Print(solution.computed_metric)
						searching := 5
						dct_matrix = do_dct(&pixel_matrix)
						dct_matrix_new = embed_to_dct(dct_matrix, "0", "Z", q)
						new_pixel_matrix = undo_dct(&dct_matrix_new)
						population := generate_population(&pixel_matrix, &new_pixel_matrix, 128, 0.9, searching)
						metric = Metric{&pixel_matrix, "0", searching, "Z", q}
						metaheuristic_info = Metaheuristic{POPULATION_SIZE, NUM_ITERATIONS, NUM_FEATURES, &metric}
						de := DE{population, metaheuristic_info, 0.3, 0.1}
						//de := SCA{population, metaheuristic_info, 2.0}
						//de := WOA{population, metaheuristic_info, float64(SEARCH_SPACE)}

						solution = de.optimize()
						for i1 := 0; i1 < 64; i1++ {
							fmt.Println(solution.result_array[i1])
						}
						ind := 0
						for i1 := block_h * 8; i1 < block_h*8+8; i1++ {
							for i2 := block_w * 8; i2 < block_w*8+8; i2++ {
								copy_img[i1][i2] -= int(solution.result_array[ind])
								ind++
							}
						}
					}
					q = generate_q(&copy_img)
				}
				imageMat := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8UC1)
				defer imageMat.Close()
				for row := 0; row < rows; row++ {
					for col := 0; col < cols; col++ {
						imageMat.SetUCharAt(row, col, uint8(copy_img[row][col]))
					}
				}
				outputFilePath := dir_path + "/saved.png"
				if ok := gocv.IMWrite(outputFilePath, imageMat); !ok {
					fmt.Println("Ошибка: не удалось сохранить изображение.")
					os.Exit(1)
				}
				fmt.Print(cnt1)
			} else if mode == "extraction" {
				q := load_q(dir_path + "/q.txt")
				var bit_string string
				var image gocv.Mat = gocv.IMRead(dir_path+"/saved.png", gocv.IMReadGrayScale)
				var rows, cols int = image.Rows(), image.Cols()
				mat_data, _ := image.DataPtrUint8()
				img := make([][]int, rows)
				for row := range img {
					img[row] = make([]int, cols)
					for col := range img[row] {
						img[row][col] = int(mat_data[row*cols+col])
					}
				}
				file, _ := os.Open(dir_path + "/blocks.txt")
				defer file.Close()
				var blocks []int
				file_scanner := bufio.NewScanner(file)
				for file_scanner.Scan() {
					line := file_scanner.Text()
					for _, num_str := range strings.Split(line, " ") {
						num, _ := strconv.Atoi(num_str)
						blocks = append(blocks, num)
					}
				}
				for _, block := range blocks {
					pixel_matrix := make([][]int, 8)
					block_w := block % (rows / 8)
					block_h := (block - block_w) / (rows / 8)
					for i1 := block_h * 8; i1 < block_h*8+8; i1++ {
						pixel_matrix[i1%8] = make([]int, 8)
						for i2 := block_w * 8; i2 < block_w*8+8; i2++ {
							pixel_matrix[i1%8][i2%8] = img[i1][i2]
						}
					}
					s := extracting_dct(&pixel_matrix, q)
					if s != "0" {
						bit_string += s[1:]
					}
					q = generate_q(&pixel_matrix)
				}
				f, _ := os.Create(dir_path + "/saved.txt")
				f.WriteString(bit_string)
				defer f.Close()
			} else if mode == "metrics" {
				var image gocv.Mat = gocv.IMRead(dir_path+"/saved.png", gocv.IMReadGrayScale)
				var rows, cols int = image.Rows(), image.Cols()
				mat_data, _ := image.DataPtrUint8()
				img := make([][]int, rows)
				for row := range img {
					img[row] = make([]int, cols)
					for col := range img[row] {
						img[row][col] = int(mat_data[row*cols+col])
					}
				}
				image = gocv.IMRead(picture, gocv.IMReadGrayScale)
				rows, cols = image.Rows(), image.Cols()
				mat_data, _ = image.DataPtrUint8()
				img_base := make([][]int, rows)
				for row := range img_base {
					img_base[row] = make([]int, cols)
					for col := range img_base[row] {
						img_base[row][col] = int(mat_data[row*cols+col])
					}
				}
				file_content, _ := os.ReadFile(dir_path + "/saved.txt")
				information := string(file_content)
				fmt.Println(metaheuristic, len(information), ssim(&img_base, &img), psnr(&img_base, &img))
			}
		}
	}
}
