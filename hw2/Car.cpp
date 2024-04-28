#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <omp.h>
#include <filesystem>
#include <chrono>

#define RGB_COMPONENT_COLOR 255

namespace fs = std::filesystem;

static const auto THREADS = std::thread::hardware_concurrency();

struct PPMPixel {
  int red;
  int green;
  int blue;
};

typedef struct {
  int x, y, all;
  PPMPixel *data;
} PPMImage;

void readPPM(const char *filename, PPMImage &img){
    std::ifstream file(filename);
    if (file) {
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s != "P3") {
            std::cout << "error in format" << std::endl;
            exit(9);
        }
        file >> img.x >> img.y;
        file >> rgb_comp_color;
        img.all = img.x * img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" << img.all
                  << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i = 0; i < img.all; i++) {
            file >> img.data[i].red >> img.data[i].green >> img.data[i].blue;
        }
    } else {
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const std::string &filename, PPMImage &img) {
    std::ofstream file(filename, std::ofstream::out);
    file << "P3" << std::endl;
    file << img.x << " " << img.y << " " << std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for (int i = 0; i < img.all; i++) {
        file << img.data[i].red << " " << img.data[i].green << " "
             << img.data[i].blue << (((i + 1) % img.x == 0) ? "n" : " ");
    }
    file.close();
}

void shiftColumns(PPMImage &img) {

    PPMPixel *temp = new PPMPixel[img.y];
    for (int i = 0; i < img.y; ++i) {
        temp[i] = img.data[i * img.x + (img.x - 1)];
    }

    #pragma omp parallel for collapse(2) num_threads(THREADS)
    for (int i = 0; i < img.y; ++i) {
        for (int j = img.x - 1; j > 0; --j) {
            img.data[i * img.x + j] = img.data[i * img.x + (j - 1)];
        }
    }

    #pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < img.y; ++i) {
        img.data[i * img.x] = temp[i];
    }

    delete[] temp;
}


int main(int argc, char *argv[]) {
    PPMImage image;
    double start, end;
    start = omp_get_wtime();

    readPPM("car.ppm", image);
    
    // Create the output directory if it doesn't exist
    fs::path outputDir = "FolderCars";
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    #pragma omp parallel for
    for (int i = 0; i < 1000; ++i) {
        shiftColumns(image);
        std::string filename = outputDir/("car_" + std::to_string(i) + ".ppm");
        writePPM(filename, image);
    }

    end = omp_get_wtime();
    double duration = end - start;
    std::cout << "Time taken: " << duration << " s" << std::endl;

    return 0;
}
