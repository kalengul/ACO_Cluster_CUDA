#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "matrix_loader.h"

#define PARAMETR_SIZE 12 // количество параметров
#define MAX_VALUE_SIZE 7 // максимальное количество значений у параметров
#define ANT_SIZE 500000 // максимальное количество значений у параметров
#define KOL_ITERATION 5000 // количество итераций ММК
#define PARAMETR_Q 1 // параметр ММК для усиления феромона Q
#define PARAMETR_RO 0.8 // параметр ММК для испарения феромона RO

//const int TABLE_SIZE = 4086; // Размер хэш-таблицы
//const int MAX_STRING_LENGTH = 256; // Максимальная длина строки в хэш-таблице
/*
struct HashEntry {
    bool occupied;
    int key; // Храним однозначное целое число
    int value; // Храним значение целевой функции
    HashEntry* next;

    HashEntry() : next(nullptr), key(-1), value(0) {}
};

// Хэш-функция

__device__ int hashFunction(int key) {
    return key % TABLE_SIZE;
}

// Функция для преобразования массива чисел в строку
__device__  void insertNumbers(int* table, int* numbers, int size) {
    int idx = threadIdx.x;

    if (idx < size) {
        int key = numbers[idx];
        int hashIndex = hashFunction(key);

        // Обработка коллизий с помощью линейного пробирования
        while (table[hashIndex] != -1) {
            hashIndex = (hashIndex + 1) % TABLE_SIZE;
        }

        table[hashIndex] = key; // Вставка значения
    }
}

// Функция для поиска в кэше
__device__ int getCachedResult(HashEntry* cache, int key) {
    int index = hashFunction(key);
    HashEntry* entry = &cache[index];

    // Проверка на наличие элементов в цепочке
    while (entry != nullptr) {
        if (entry->occupied && entry->key == key) {
            return entry->value; // Найдено
        }
        entry = entry->next; // Переход к следующему элементу в цепочке
    }
    return -1; // Не найдено (можно использовать -1 как индикатор отсутствия значения)
}


// Функция для сохранения результата в кэше
__device__ void saveToCache(HashEntry* cache, int key, int value) {
    int index = hashFunction(key);
    HashEntry* newEntry = new HashEntry(); // Выделяем память для нового элемента
    newEntry->key = key;
    newEntry->value = value;

    // Добавляем в начало цепочки
    if (!cache[index].occupied) {
        cache[index] = *newEntry; // Если ячейка пуста, добавляем новый элемент
    }
    else {
        HashEntry* current = &cache[index];
        while (current->next != nullptr) {
            current = current->next; // Находим конец цепочки
        }
        current->next = newEntry; // Добавляем новый элемент в конец цепочки
    }
}
*/




// Функция для цвычисления параметра х1 при 12 параметрическом графе
__device__ double go_x1_6(double* parametr) {
    return parametr[0] * (parametr[1] + parametr[2] + parametr[3] + parametr[4] + parametr[5]);
}

// Функция для цвычисления параметра х2 при 12 параметрическом графе
__device__ double go_x2_6(double* parametr) {
    return parametr[6] * (parametr[7] + parametr[8] + parametr[9] + parametr[10] + parametr[11]);
}

// Функция для целевой функции Шаффера
__device__ double BenchShafferaFunction(double* parametr) {
    double x1 = go_x1_6(parametr);
    double x2 = go_x2_6(parametr);
    double r = sqrt(x1 * x1 + x2 * x2);
    double sin_r = sin(r);
    return 1.0 / 2.0 - (sin_r * sin_r - 0.5) / (1.0 + 0.001 * (x1 * x1 + x2 * x2));
}

// Функция для вычисления вероятностной формулы
// Входные данные - значение феромона pheromon и количества посещений вершины kol_enter
__device__ double probability_formula(double pheromon, double kol_enter) {
     double res = 0;
     if (kol_enter != 0) {
         res = 1.0 / kol_enter + pheromon;
     }
    return res;
}
//Подготовка массива для вероятностного поиска
//pheromon,kol_enter - матрицы слоев, если надо больше придется менять
//norm_matrix_probability -  итоговая отнормированная матрица
//probability_formula - функция для вычисления вероятностной формулы
__global__ void go_mass_probability(double* pheromon, double* kol_enter, double* norm_matrix_probability) {
//    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (столбца)
    
    //Нормализация слоя с феромоном
    double sumVector=0;
    double pheromon_norm[MAX_VALUE_SIZE];
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        sumVector += pheromon[MAX_VALUE_SIZE * tx + i];
    }
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        pheromon_norm[i] = pheromon[MAX_VALUE_SIZE * tx + i] / sumVector;
    }
    sumVector = 0;
    double svertka[MAX_VALUE_SIZE];
    
    for (int i = 0; i < MAX_VALUE_SIZE; i++) {
        svertka[i] = probability_formula(pheromon_norm[i], kol_enter[MAX_VALUE_SIZE * tx + i]);
        sumVector += svertka[i];
    }
    
    norm_matrix_probability[MAX_VALUE_SIZE * tx] = (svertka[0]) / sumVector;
    for (int i = 1; i < MAX_VALUE_SIZE; i++) {
        norm_matrix_probability[MAX_VALUE_SIZE * tx + i] = (svertka[i]) / sumVector + norm_matrix_probability[MAX_VALUE_SIZE * tx + i-1]; //Нормаирование значений матрицы с накоплением
    }
}

//Вычисление пути агентов
// parametr - матрица с значениями параметров для вычисления х1 и х2
// norm_matrix_probability - нормализованная матрица вероятностей выбора вершины
// cache - хэш-таблица (ПОКА НЕ РАБОТАЕТ)
// agent - значения параметров для агента (СКОРЕЕ ВСЕГО НЕ НУЖНО ВООБЩЕ)
// agent_node - номер значения для каждого параметра для пути агента
// OF - значения целевой функции для агента
__global__ void go_all_agent(int* gpuTime, double* parametr, double* norm_matrix_probability, double* agent, double* agent_node, double* OF) {
    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (столбца)
    int seed = 123 + bx * ANT_SIZE + tx * PARAMETR_SIZE +gpuTime[0];
        
    // Генерация случайного числа с использованием curand
    curandState state;
    curand_init(seed, 0, 0, &state); // Инициализация состояния генератора случайных чисел
    double randomValue = curand_uniform(&state) * 1; // Генерация случайного числа в диапазоне [0, 1]

    //Определение номера значения
    int k = 0;
    while (k < PARAMETR_SIZE && randomValue > norm_matrix_probability[MAX_VALUE_SIZE * tx + k]) {
        k++;
    }
    // Запись подматрицы блока в глобальную память
    // каждый поток записывает один элемент
    agent_node[bx * PARAMETR_SIZE + tx] = k;
    agent[bx * PARAMETR_SIZE + tx] = parametr[tx * MAX_VALUE_SIZE + k];
    OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
    /*
    double cachedResult = 0;//getCachedResult(cache, 0, 0);
    if (cachedResult != 0) {
        OF[bx] = cachedResult; // Используем закешированное значение
    }
    else {
        OF[bx] = BenchShafferaFunction(&agent[bx * PARAMETR_SIZE]);
        //saveToCache(cache, 0, 0, OF[bx]);
    }*/
}

//Обновление слоев графа
// pheromon - слой с весами (феромоном)
// kol_enter - слой с количеством посещений вершины
// agent_node - пути агентов
// OF - значение целевой функции для каждого агента
__global__ void add_pheromon_iteration(double* pheromon, double* kol_enter, double* agent_node, double* OF) {
//    int bx = blockIdx.x; // индекс блока (не требуется)
    int tx = threadIdx.x; // индекс потока (параметра)
    //Испарение весов-феромона
    for (int i = 0; i < MAX_VALUE_SIZE; ++i) {
        pheromon[MAX_VALUE_SIZE * tx + i] = pheromon[MAX_VALUE_SIZE * tx + i] * PARAMETR_RO;
    }
    //Добавление весов-феромона
    for (int i = 0; i < ANT_SIZE; ++i) {
        int k = int(agent_node[i * PARAMETR_SIZE + tx]);
        kol_enter[MAX_VALUE_SIZE * tx + k]++;
        pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q * OF[i]; //MAX
//        pheromon[MAX_VALUE_SIZE * tx + k] = pheromon[MAX_VALUE_SIZE * tx + k] + PARAMETR_Q / OF[i]; //MIN
    }
//        for (int i = 0; i < PARAMETR_SIZE; ++i) {
//           kol_enter[MAX_VALUE_SIZE * i + int(agent_node[tx * PARAMETR_SIZE + i])]++;
}

// Функция для загрузки матрицы из файла
bool load_matrix(const std::string& filename, double* parametr_value, double* pheromon_value, double* kol_enter_value) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Don't open file!" << std::endl;
        return false;
    }

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> parametr_value[k])) { // Чтение элемента в массив a
                std::cerr << "Error load element [" << i << "][" << j << "]" << std::endl;
                return false;
            }
            
            if (parametr_value[k] != -100) {
                pheromon_value[k] = 1.0; // Присваиваем значение pheromon_value
                kol_enter_value[k] = 1.0;
            }
            else {
                pheromon_value[k] = 0.0; // Присваиваем значение pheromon_value
                parametr_value[k] = 0.0; //Нужно ли????
                kol_enter_value[k] = 0.0;
            }

            
        }
    }
    infile.close();
    return true;
}

int main(int argc, char* argv[]) {
    int numBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE * sizeof(double);
    int kolBytes_matrix_graph = MAX_VALUE_SIZE * PARAMETR_SIZE;
    int numBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE * sizeof(double);
    int kolBytes_matrix_ant = PARAMETR_SIZE * ANT_SIZE;
    int numBytes_ant = ANT_SIZE * sizeof(double);
    double global_maxOf = -INT16_MAX;
    double global_minOf = INT16_MAX;

    // Выделение памяти на хосте
    double* parametr_value = new double[kolBytes_matrix_graph];
    double* pheromon_value = new double[kolBytes_matrix_graph];
    double* kol_enter_value = new double[kolBytes_matrix_graph];
    double* norm_matrix_probability = new double[kolBytes_matrix_graph];
    double* ant = new double[kolBytes_matrix_ant];
    double* ant_parametr = new double[kolBytes_matrix_ant];
    double* antOF = new double[ANT_SIZE];
    double* antSumOF = new double[ANT_SIZE];

    load_matrix("Parametr_Graph/test.txt", parametr_value, pheromon_value, kol_enter_value);

    // Выделение памяти на устройстве
    double* parametr_value_dev = nullptr;
    double* pheromon_value_dev = nullptr;
    double* kol_enter_value_dev = nullptr;
    double* antdev = nullptr;
    double* norm_matrix_probability_dev = nullptr;
    double* antOFdev = nullptr;
    double* ant_parametr_dev = nullptr;
    int* gpuTime_dev = nullptr;
   // HashEntry* cache_dev = nullptr;
    

    cudaMalloc((void**)&parametr_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&pheromon_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&kol_enter_value_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&antdev, numBytes_matrix_ant);
    cudaMalloc((void**)&norm_matrix_probability_dev, numBytes_matrix_graph);
    cudaMalloc((void**)&antOFdev, numBytes_ant);
    cudaMalloc((void**)&ant_parametr_dev, numBytes_matrix_ant);
    //cudaMalloc((void**)&cache_dev, TABLE_SIZE * sizeof(HashEntry));
    cudaMalloc((void**)&gpuTime_dev, sizeof(int));
    /*
    // Инициализация хэш-таблицы
    HashEntry* h_cache = new HashEntry[TABLE_SIZE];
    for (int i = 0; i < TABLE_SIZE; ++i) {
        h_cache[i].occupied = false; // Устанавливаем все ячейки как не занятые
        h_cache[i].next = nullptr; // Инициализируем указатели на следующий элемент
    }
    cudaMemcpy(cache_dev, h_cache, TABLE_SIZE * sizeof(HashEntry), cudaMemcpyHostToDevice);
    delete[] h_cache;
    */
    // Установка конфигурации запуска ядра
    dim3 kol_parametr(PARAMETR_SIZE);
    dim3 kol_ant (ANT_SIZE);

    // Создание обработчиков событий CUDA
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    int i_gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Асинхронно выдаем работу на GPU (все в поток 0)
    cudaEventRecord(start, 0);
    cudaMemcpy(parametr_value_dev, parametr_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(pheromon_value_dev, pheromon_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    cudaMemcpy(kol_enter_value_dev, kol_enter_value, numBytes_matrix_graph, cudaMemcpyHostToDevice);
    for (int nom_iter = 0; nom_iter < KOL_ITERATION; ++nom_iter) {

        cudaMemcpy(gpuTime_dev, &i_gpuTime, sizeof(int), cudaMemcpyHostToDevice);
        /*
        std::cout << "Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }
        */

        go_mass_probability <<<1, kol_parametr >>> (pheromon_value_dev, kol_enter_value_dev, norm_matrix_probability_dev);
        go_all_agent << <kol_ant, kol_parametr >> > (gpuTime_dev, parametr_value_dev, norm_matrix_probability_dev, antdev, ant_parametr_dev, antOFdev);
        add_pheromon_iteration << <1, kol_ant >> > (pheromon_value_dev, kol_enter_value_dev, ant_parametr_dev, antOFdev);

        cudaMemcpy(norm_matrix_probability, norm_matrix_probability_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
        cudaMemcpy(ant, antdev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
        cudaMemcpy(ant_parametr, ant_parametr_dev, numBytes_matrix_ant, cudaMemcpyDeviceToHost);
        cudaMemcpy(antOF, antOFdev, numBytes_ant, cudaMemcpyDeviceToHost);

        cudaMemcpy(pheromon_value, pheromon_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
        cudaMemcpy(kol_enter_value, kol_enter_value_dev, numBytes_matrix_graph, cudaMemcpyDeviceToHost);
        // Копируйте данные seeds обратно на хост

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        i_gpuTime=int(gpuTime);
        /*
        std::cout << "norm_matrix_probability (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << norm_matrix_probability[i * MAX_VALUE_SIZE + j] << " "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }
        */
        double maxOf = -INT16_MAX;
        double minOf = INT16_MAX;
        //std::cout << "h_seeds (" << int(gpuTime) << "x" << ANT_SIZE << "):" << std::endl;
        for (int i = 0; i < ANT_SIZE; ++i) {
            /*for (int j = 0; j < PARAMETR_SIZE; ++j) {
                std::cout << ant_parametr[i * PARAMETR_SIZE + j];// << "(" << ant[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << "-> " << antOF[i] << std::endl;
            */
            if (antOF[i] > maxOf) {
                maxOf = antOF[i];
            }
            if (antOF[i] < minOf) {
                minOf = antOF[i];
            }
        }
        
        /*
        std::cout << "New Matrix (" << MAX_VALUE_SIZE << "x" << PARAMETR_SIZE << "):" << std::endl;
        for (int i = 0; i < PARAMETR_SIZE; ++i) {
            for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
                std::cout << parametr_value[i * MAX_VALUE_SIZE + j] << "(" << pheromon_value[i * MAX_VALUE_SIZE + j] << ", " << kol_enter_value[i * MAX_VALUE_SIZE + j] << ") "; // Индексируем элементы
            }
            std::cout << std::endl; // Переход на новую строку
        }     

        */
        
        if (minOf < global_minOf) {
            global_minOf = minOf;
        }
        if (maxOf > global_maxOf) {
            global_maxOf = maxOf;
        }
        std::cout << "MIN OF -> " << minOf << "  MAX OF -> " << maxOf << " GMIN OF -> " << global_minOf << "  GMAX OF -> " << global_maxOf << " Time: " << gpuTime << " ms " << std::endl;

    }


    // Освобождение ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(parametr_value_dev);
    cudaFree(pheromon_value_dev);
    cudaFree(kol_enter_value_dev);
    cudaFree(norm_matrix_probability_dev);
    cudaFree(antdev);
    cudaFree(ant_parametr_dev);
    cudaFree(antOFdev);

    delete[] parametr_value;
    delete[] pheromon_value;
    delete[] kol_enter_value;
    delete[] norm_matrix_probability;
    delete[] ant;
    delete[] ant_parametr;
    delete[] antOF;

    return 0;
}
