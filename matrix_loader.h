#pragma once
#ifndef MATRIX_LOADER_H
#define MATRIX_LOADER_H

#include <iostream>
#include <fstream>
#include <string>

const int PARAMETR_SIZE = 10; // Задайте нужный размер
const int MAX_VALUE_SIZE = 10; // Задайте нужный размер

extern "C" {
    bool load_matrix(const std::string& filename, float* a, float* b);
}

#endif // MATRIX_LOADER_H
