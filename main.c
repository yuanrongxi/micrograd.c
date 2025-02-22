#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "micrograd.h"

#define BMP_TYPE 0x4D42
#define BMP_HEADER_SIZE 54
#define COLOR_TAB_N 256

#pragma pack(push, 1)
typedef struct {
    uint16_t type;      // File type
    uint32_t size;      // File size in bytes
    uint16_t reserved1; // Reserved
    uint16_t reserved2; // Reserved
    uint32_t off_bits;   // Offset to pixel data
} bmp_file_header_t;

// BMP Info Header
typedef struct {
    uint32_t size;          // Info header size
    int32_t  width;         // Image width
    int32_t  height;        // Image height
    uint16_t planes;        // Number of color planes
    uint16_t bit_count;      // Bits per pixel
    uint32_t compression;   // Compression type
    uint32_t size_image;     // Image size in bytes
    int32_t  x_pels_meter; // Pixels per meter (X)
    int32_t  y_pels_meter; // Pixels per meter (Y)
    uint32_t clr_used;       // Number of colors used
    uint32_t clr_important;  // Important colors
} bmp_info_header_t;

// BMP Color Table Entry
typedef struct {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t reserved;
} bmp_color_table_t;
#pragma pack(pop)


static inline void map_to_point(yinyang_sample_t* sample, uint8_t* pixels, int width, int height) {
    int x = (sample->pos[0] - min_x) * width / (max_x - min_x);
    int y = (max_y - sample->pos[1]) * height / (max_y - min_y);

    if(x >= 0 && x < width && y >= 0 && y < height) {
        pixels[y * width + x] = (uint8_t)(sample->label == 0 ? 0: (sample->label == 1 ? 128 : 196));
    }
}

static void yinyang_to_pixels(int n, uint8_t* pixels, int width, int height) {
    yinyang_sample_t* samples = malloc(n * sizeof(yinyang_sample_t));
    random_t r;

    random_init(&r, 42);
    yinyang_gen_data(samples, &r, 0.5, 0.1, n);

    for(int i = 0; i < n; i++) {
        map_to_point(samples + i, pixels, width, height);
    }
     free(samples);
}

void generate_yinyang_bmp(const char* filename, int width, int height, int n) {
    bmp_file_header_t header;
    bmp_info_header_t info;
    bmp_color_table_t color_table[COLOR_TAB_N];
    for(int i = 0; i < COLOR_TAB_N; i++) {
        color_table[i].blue = i;
        color_table[i].green = i;
        color_table[i].red = i;
        color_table[i].reserved = 0;
    }

    header.type = BMP_TYPE; // "BM"
    header.off_bits = BMP_HEADER_SIZE + COLOR_TAB_N * sizeof(bmp_color_table_t);
    header.size = header.off_bits + width * height;
    header.reserved1 = 0;
    header.reserved2 = 0;

    info.size = sizeof(bmp_info_header_t);
    info.width = width;
    info.height = height;
    info.planes = 1;
    info.bit_count = 8; // 8 bits per pixel
    info.compression = 0; // BI_RGB
    info.size_image = width * height;
    info.x_pels_meter = 0; // 72 DPI
    info.y_pels_meter = 0; // 72 DPI
    info.clr_used = 256;
    info.clr_important = 256;

    FILE* file = fopen(filename, "wb");
    if(!file) {
        printf("Failed to open file %s\n", filename);
        return;
    }

    fwrite(&header, sizeof(bmp_file_header_t), 1, file);
    fwrite(&info, sizeof(bmp_info_header_t), 1, file);
    fwrite(color_table, sizeof(bmp_color_table_t), COLOR_TAB_N, file);

    uint8_t* pixels = (uint8_t*)malloc(width * height);
    memset(pixels, 255, width * height);

    yinyang_to_pixels(n, pixels, width, height);

    fwrite(pixels, width * height, 1, file);

    free(pixels);
    fclose(file);
    printf("Generated %s successfully\n", filename);
}


int main(int argc, const char* argv[]) {
    //generate_yinyang_bmp("yinyang.bmp", 512, 512, 200000);
    //test_micrograd();
    training();
    return 0;
}
