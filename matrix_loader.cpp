#include "matrix_loader.h"

bool load_matrix(const std::string& filename, float* a, float* b) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Не удалось открыть файл!" << std::endl;
        return false;
    }

    for (int i = 0; i < PARAMETR_SIZE; ++i) {
        for (int j = 0; j < MAX_VALUE_SIZE; ++j) {
            int k = MAX_VALUE_SIZE * i + j;
            if (!(infile >> a[k])) { // Чтение элемента в массив a
                std::cerr << "Ошибка чтения элемента [" << i << "][" << j << "]" << std::endl;
                return false;
            }
            b[k] = 10.0f; // Присваиваем значение b
        }
    }

    infile.close();
    return true;
}
