#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>
#include <assert.h>

typedef struct {
    uint64_t seed;
} random_t;

void random_init(random_t* ra, uint64_t seed);
float unirandom(random_t* ra);
float uniform(random_t* ra, float min, float max);

#define POS_SIZE 2
typedef struct {
    float pos[POS_SIZE];
    int label;
} yinyang_sample_t;

#define max_x 2
#define max_y 2
#define min_x -2
#define min_y -2
void yinyang_gen_data(yinyang_sample_t* samples, random_t* ra, float big, float small, int n);


typedef struct node_t{
    void* key;
    struct node_t* next;
}node_t;

typedef struct {
    size_t size;
    size_t align_size; //object alignment size
    int n;
    node_t* head;
    node_t* buckets[];
} table_t;

table_t* table_new(size_t size, size_t obj_size);
void table_insert(table_t* ht, const void* key);
int table_has(table_t* ht, const void* key);
void table_clean(table_t* ht);
void table_free(table_t* ht);

//slab and gc
void* slab_alloc(size_t size);
void slab_free(void* ptr);
void free_all_slabs();

void* push_gc(void* ptr);
void clean_gc();  

int64_t get_time_us();

#endif
