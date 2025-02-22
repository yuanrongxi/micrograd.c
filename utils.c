#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "utils.h"


#ifndef sqr
static inline float sqr(float x) {
    return x * x;
}
#endif

void random_init(random_t *ra, uint64_t seed) {
    ra->seed = seed;
}

static inline uint32_t random_u32(random_t* ra) {
    ra->seed ^= (ra->seed << 12) & 0xFFFFFFFFFFFFFFFF;
    ra->seed ^= (ra->seed >> 25) & 0xFFFFFFFFFFFFFFFF;
    ra->seed ^= (ra->seed << 27) & 0xFFFFFFFFFFFFFFFF;
    return ((ra->seed * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF;
}

float unirandom(random_t* ra) {
    return (float)(random_u32(ra) >> 8) / 16777216.0f;
}

float uniform(random_t* ra, float min, float max) {
    return min + (max - min) * unirandom(ra);
}

static inline float right_dot_dist(float x, float y, float r) {
    return sqrt(sqr(x - 1.5 * r) + sqr(y - r));
}

static inline float left_dot_dist(float x, float y, float r) {
    return sqrt(sqr(x - 0.5 * r) + sqr(y - r));
}

static inline int which_class(float x, float y, float big_r, float small_r) {
    float d_right = right_dot_dist(x, y, big_r);
    float d_left = left_dot_dist(x, y, big_r);

    //circles
    if (d_right <= small_r || d_left <= small_r) {
        return 2;
    }
    //yin
    if ((d_right > big_r / 2 && y > big_r) || (d_left < big_r / 2 && d_left > small_r)) {
        return 0;
    }
    //yang
    return 1;
}

static void gen_sample(yinyang_sample_t* sample, float big, float small, int class, random_t* ra){
    for(;;){
        float x = uniform(ra, 0, big * 2);
        float y = uniform(ra, 0, big * 2);
        if (sqrt(sqr(x - big) + sqr(y - big)) > big){
            continue;
        }

        sample->label = which_class(x, y, big, small);
        if(sample->label == class){
            sample->pos[0] = (x/big - 1) * max_x;
            sample->pos[1] = (y/big - 1) * max_y;
            break;
        }
    }
}

void yinyang_gen_data(yinyang_sample_t* samples, random_t* ra, float big, float small, int n) {
    yinyang_sample_t* s = samples;
    for(int i = 0; i < n; ++i) {
        gen_sample(s, big, small, i % 3, ra);
        s++;
    }  
}

static inline size_t get_index(const void* ptr, size_t size, size_t align_size) {
    uintptr_t ptr_val = (uintptr_t)ptr;
    return (ptr_val / align_size) % size;
}

//memory alignment value 0x10   
#define PTR_ALIGN 0x10
table_t* table_new(size_t size, size_t obj_size){
    table_t* ht = (table_t*)malloc(sizeof(table_t) + size * sizeof(node_t*));
    ht->n = 0;
    ht->size = size;
    ht->head = NULL;
    ht->align_size = ((obj_size + PTR_ALIGN - 1) / PTR_ALIGN) * PTR_ALIGN;

    for(int i = 0; i < size; ++i){
        ht->buckets[i] = NULL;
    }
    return ht;
}

void table_free(table_t* ht){
    if(ht == NULL)
        return;

    for(int i = 0; i < ht->size; ++i){
        node_t* node = ht->buckets[i];
        while(node != NULL){
            node_t* next = node->next;
            free(node);
            node = next;
        }
    }

    while(ht->head != NULL){
        node_t* node = ht->head;
        ht->head = ht->head->next;
        free(node);
    }
    free(ht);
}

void table_insert(table_t* ht, const void* key){
    if(ht == NULL || key == NULL)
        return;

    //check if key exists
    size_t index = get_index(key, ht->size, ht->align_size);
    node_t* node = ht->buckets[index];
    while(node != NULL){
        if(node->key == key)
            return;
        node = node->next;
    }   

    //get node from head
    if(ht->head == NULL){
        node = (node_t*)malloc(sizeof(node_t));
    }else{
        node = ht->head;
        ht->head = node->next;
    }

    //insert node
    node->key = (void*)key;
    node->next = ht->buckets[index];
    ht->buckets[index] = node;

    ht->n++;
}

int table_has(table_t* ht, const void* key){
    if(ht == NULL || key == NULL)
        return 0;

    size_t index = get_index(key, ht->size, ht->align_size);

    node_t* node = ht->buckets[index];
    while(node != NULL){
        if(node->key == key)
            return 1;
        node = node->next;
    }

    return 0;
}

void table_clean(table_t* ht){
    if(ht == NULL)
        return;

    for(int i = 0; i < ht->size; ++i){
        node_t* node = ht->buckets[i];
        while(node != NULL){
            node_t* next = node->next;
            node->next = ht->head;
            ht->head = node;

            node = next;
        }
        ht->buckets[i] = NULL;
    }
    ht->n = 0;
}

#define SLAB_MAGIC 0xdeff0110
#define SLAB_PTR_SIZE sizeof(slab_t)
#define SLAB_PTR(ptr) ((slab_t*)(((uintptr_t)ptr) - SLAB_PTR_SIZE))

struct slab_s;
typedef struct slab_s slab_t;
struct slab_s {
    int magic;
    slab_t* next;
    uint8_t data[];
};

//gc slab
typedef struct {
    slab_t* head;   //waiting for free slabs
    slab_t* used;   //used slabs
    uint32_t alloc_count;
    uint32_t free_count;
}gc_t;

gc_t gc = {NULL, NULL, 0, 0};

void* slab_alloc(size_t size) {
    slab_t* slab;
    if (gc.head == NULL) {
        slab = malloc(sizeof(slab_t) + size);
        slab->magic = SLAB_MAGIC;
        slab->next = NULL;
        
        gc.alloc_count++;
    } else {
        slab = gc.head;
        gc.head = gc.head->next;
    }

    return slab->data;
}

void slab_free(void* ptr) {
    if(ptr == NULL)
        return;

    slab_t* slab = SLAB_PTR(ptr);
    if(slab->magic == SLAB_MAGIC) {
        slab->next = gc.head;   
        gc.head = slab;
    }  
}

//free all slabs
void free_all_slabs() {
    while(gc.head != NULL) {
        slab_t* slab = gc.head;
        gc.head = gc.head->next;
        free(slab);

        gc.free_count++;
    }
    printf("gc alloc: %u, free: %u\n", gc.alloc_count, gc.free_count);
}

void* push_gc(void* ptr) {
    if(ptr != NULL) {
        slab_t* slab = SLAB_PTR(ptr);
        assert(slab->magic == SLAB_MAGIC);
        slab->next = gc.used;
        gc.used = slab;
    }
    return ptr;
}

void clean_gc() {
    while(gc.used != NULL) {
        slab_t* slab = gc.used;
        assert(slab->magic == SLAB_MAGIC);
        gc.used = slab->next;

        //push to head waiting for free
        slab->next = gc.head;
        gc.head = slab;
    }
}

int64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}