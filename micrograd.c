#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "micrograd.h"

#define MAX_IN 2
#define MAX_OP_LEN 8

struct value_s;
typedef struct value_s value_t;

// backprop function type
typedef void(*backward_func) (value_t* out, value_t* in1, value_t* in2);

// value object stores a single scalar value and its gradient
struct value_s {
    float v;                //value of scalar
    float grad;             //gradient
    float f;                // first moment
    float s;                // second moment
    backward_func backward; //backprop function

    value_t* in[MAX_IN];    // previous values
    value_t* next;          // next in the backpropagation topo

    char op[MAX_OP_LEN];    //operation
};

//simulate gc new
static value_t* value_gc_new(float v, const char* op, value_t* in1, value_t* in2, backward_func backward) {
    value_t* out = slab_alloc(sizeof(value_t));
    out->v = v;
    out->grad = 0.0f;
    out->f = 0.0f;
    out->s = 0.0f;
    out->backward = backward;
    out->in[0] = in1;
    out->in[1] = in2;
    out->next = NULL;

    strncpy(out->op, op, MAX_OP_LEN);

    //add to gc
    push_gc(out);

    return out;
}

static value_t* set_value(value_t* value, float v, const char* op, value_t* in1, value_t* in2) {
    value->v = v;
    value->in[0] = in1;
    value->in[1] = in2;
    value->backward = NULL;
    value->next = NULL;
    value->grad = 0.0f;

    strncpy(value->op, op, MAX_OP_LEN);
    return value;  
}

static value_t* value_new(float v) {
    value_t* value = slab_alloc(sizeof(value_t));
    value->v = v;
    value->grad = 0.0f;
    value->f = 0.0f;
    value->s = 0.0f;
    value->backward = NULL;
    value->in[0] = NULL;
    value->in[1] = NULL;
    value->next = NULL;
    return value;
}

static void value_free(value_t* value) {
    slab_free(value);
}

static void add_backward(value_t* out, value_t* in1, value_t* in2) {
    in1->grad += out->grad;
    in2->grad += out->grad;
}

static value_t* value_add(value_t* in1, value_t* in2) {
    return value_gc_new(in1->v + in2->v, "+", in1, in2, add_backward);
}

static void* value_add_v(value_t* in, float v) {
    value_t* c = value_gc_new(v, "+", NULL, NULL, NULL);
    return value_add(in, c);
}

static void sub_backward(value_t* out, value_t* in1, value_t* in2) {
    in1->grad += out->grad;
    in2->grad -= out->grad;
}

static value_t* value_sub(value_t* in1, value_t* in2) {
    return value_gc_new(in1->v - in2->v, "-", in1, in2, sub_backward);
}

static void* value_sub_v(value_t* in, float v) {
    value_t* c = value_gc_new(v, "-", NULL, NULL, NULL);
    return value_sub(in, c);
}

static void mul_backward(value_t* out, value_t* in1, value_t* in2) {
    in1->grad += out->grad * in2->v;
    in2->grad += out->grad * in1->v;
}

static value_t* value_mul(value_t* in1, value_t* in2) {
    return value_gc_new(in1->v * in2->v, "*", in1, in2, mul_backward);
}

static void* value_mul_v(value_t* in, float v) {
    value_t* c = value_gc_new(v, "*", NULL, NULL, NULL);
    return value_mul(in, c);
}

static void div_backward(value_t* out, value_t* in1, value_t* in2) {
    in1->grad += out->grad / in2->v;
    in2->grad -= out->grad * in1->v / (in2->v * in2->v);
}

static value_t* value_div(value_t* in1, value_t* in2) {
    return value_gc_new(in1->v / in2->v, "/", in1, in2, div_backward);
}

static void* value_div_v(value_t* in, float v) {
    value_t* c = value_gc_new(v, "/", NULL, NULL, NULL);
    return value_div(in, c);
}

static value_t* value_neg(value_t* in) {
    return value_mul_v(in, -1.0f);
}

static void pow_backward(value_t* out, value_t* in, value_t* exp) {
    in->grad += out->grad * exp->v * pow(in->v, exp->v - 1);
}

static value_t* value_pow(value_t* in, value_t* exp) {
    return value_gc_new(pow(in->v, exp->v), "pow", in, exp, pow_backward);
}

static void* value_pow_v(value_t* in, float v) {
    value_t* c = value_gc_new(v, "pow", NULL, NULL, NULL);
    return value_pow(in, c);
}

static void exp_backward(value_t* out, value_t* in, value_t* in2) {
    in->grad += out->grad * out->v;
}

static value_t* value_exp(value_t* in) {
    return value_gc_new(exp(in->v), "exp", in, NULL, exp_backward);
}

static void tanh_backward(value_t* out, value_t* in,  value_t* in2) {
    in->grad += out->grad * (1 - pow(tanh(in->v), 2));
}

static value_t* value_tanh(value_t* in) {
    return value_gc_new(tanh(in->v), "tanh", in, NULL, tanh_backward);
}

static void relu_backward(value_t* out, value_t* in,  value_t* in2) {
    in->grad += out->grad * (in->v > 0 ? 1 : 0);
}

static value_t* value_relu(value_t* in) {
    return value_gc_new(in->v > 0 ? in->v : 0, "relu", in, NULL, relu_backward);
}

static void log_backward(value_t* out, value_t* in,  value_t* in2) {
    in->grad += out->grad / in->v;
}

static value_t* value_log(value_t* in) {
    return value_gc_new(log(in->v), "log", in, NULL, log_backward);
}

static value_t* build_topo(value_t* val, value_t* topo, table_t* visited) {
    if(table_has(visited, val))
        return topo;

    for (int i = 0; i < MAX_IN; i++) {
        if (val->in[i] != NULL) {
            topo = build_topo(val->in[i], topo, visited);
        }
    }
    val->next = topo;

    table_insert(visited, val);

    return val;
}

static void backward(value_t* out, table_t* visited) {
    //build the topo of the backpropagation graph
    value_t* topo = build_topo(out, NULL, visited);

    //set the gradient of the output to 1.0f
    out->grad = 1.0f;
    //backpropagate the gradient
    for (value_t* v = topo; v != NULL; v = v->next) {
        if(v->backward != NULL) {
            v->backward(v, v->in[0], v->in[1]);
        }
    }

    //clear the values of the visited table from this round of backward
    table_clean(visited);
}

static void print_value(value_t* val) {
    
    printf("Value: %f, Grad: %f, Op: %s, in1: %f, in2: %f\n", val->v, val->grad, val->op, 
        val->in[0] != NULL ? val->in[0]->v : 0, 
        val->in[1] != NULL ? val->in[1]->v : 0);
}

static void print_topo(value_t* topo) {
    for (value_t* v = topo; v != NULL; v = v->next) {
        print_value(v);
    }
}

//adamw optimizer
typedef struct {
    int t;          //time step
    float lr;       //learning rate
    float beta1;    //first moment
    float beta2;    //second moment
    float eps;      //epsilon
    float wd;       //weight decay
} adamw_t;

static void adamw_step(value_t* v, adamw_t* optimizer) {
    v->f = optimizer->beta1 * v->f + (1 - optimizer->beta1) * v->grad;
    v->s = optimizer->beta2 * v->s + (1 - optimizer->beta2) * v->grad * v->grad;
    float m_f = v->f / (1 - pow(optimizer->beta1, optimizer->t));
    float m_s = v->s / (1 - pow(optimizer->beta2, optimizer->t));

   v->v -= optimizer->lr * (m_f / (sqrt(m_s) + optimizer->eps) + optimizer->wd * v->v);
}

//neuron
typedef struct {
    value_t* w;      //weights
    value_t* b;      //bias
    int w_size;      //weights size
    int nonlin;      //non-linearity
}neuron_t;

static random_t rng;

static neuron_t* neuron_new(int nin, int nonlin) {
    neuron_t* neuron = malloc(sizeof(neuron_t) + nin * sizeof(value_t) + sizeof(value_t));
    neuron->w = (value_t*)(neuron + 1);
    neuron->b = (value_t*)(neuron->w + nin);
    neuron->w_size = nin;   
    neuron->nonlin = nonlin;

    //initialize weights and bias
    value_t* w = neuron->w;
    for(int i = 0; i < nin; i++) {
        set_value(w++, uniform(&rng, -1, 1) * sqrt(2.0f / nin), "", NULL, NULL);
    }
    set_value(neuron->b, 0.0f, "", NULL, NULL);

    return neuron;
}

static void neuron_free(neuron_t* neuron) {
    free(neuron);
}   

static value_t* neuron_forward(neuron_t* neuron, value_t** x) {
    value_t* out = value_gc_new(0, "", NULL, NULL, NULL);
    value_t* w = neuron->w;

    //y = w1 * x1 + w2 * x2 + ... + wn * xn + b, 
    for (int i = 0; i < neuron->w_size; i++) {
        out = value_add(out, value_mul(w++, x[i]));
    }
    out = value_add(out, neuron->b);

    return (neuron->nonlin == 1) ? value_tanh(out) : out;
}

static void neuron_step(neuron_t* neuron, adamw_t* optimizer) {
    value_t* w = neuron->w;
    for (size_t i = 0; i < neuron->w_size; i++){
        adamw_step(w++, optimizer);
    }
    adamw_step(neuron->b, optimizer);
}

static void zero_grad(neuron_t* neuron) {
    value_t* w = neuron->w;
    for (size_t i = 0; i < neuron->w_size; i++){
        w->grad = 0;
        w++;
    }
    neuron->b->grad = 0;
}

static int neuron_size(neuron_t* neuron) {
    return neuron->w_size * 2 + 1 + ((neuron->nonlin == 1) ? 1 : 0);
}

typedef struct {
    neuron_t** neurons; //neurons
    value_t** out;      //output's values array, size = n
    int n;              //number of neurons
}layer_t;

static layer_t* layer_new(int nin, int nout, int nonlin) {
    layer_t* layer = malloc(sizeof(layer_t) + nout * sizeof(neuron_t**) + nout * sizeof(value_t**));
    layer->neurons = (neuron_t**)(layer + 1);
    layer->out = (value_t**)(layer->neurons + nout);
    layer->n = nout;

    for (int i = 0; i < nout; i++) {
        layer->neurons[i] = neuron_new(nin, nonlin);
    }

    return layer;
}

static void layer_free(layer_t* layer) {
    if(layer != NULL) {
        for (int i = 0; i < layer->n; i++) {
            neuron_free(layer->neurons[i]);
        }
        free(layer);
    }
}

static value_t** layer_forward(layer_t* layer, value_t** x) {
    for (int i = 0; i < layer->n; i++) {
        layer->out[i] = neuron_forward(layer->neurons[i], x);
    }
    return layer->out;
}

static void layer_step(layer_t* layer, adamw_t* optimizer) {
    for (int i = 0; i < layer->n; i++) {
        neuron_step(layer->neurons[i], optimizer);
    }
}

static void layer_zero_grad(layer_t* layer) {
    for (int i = 0; i < layer->n; i++) {
        zero_grad(layer->neurons[i]);
    }
}

static int layer_size(layer_t* layer) {
    int size = 0;
    for (int i = 0; i < layer->n; i++) {
        size += neuron_size(layer->neurons[i]);
    }
    return size;
}

//multi-layer perceptron
typedef struct {
    layer_t** layers; //layers
    int nin;          //input size
    int nout;         //output size
    int n;            //number of layers
}mpl_t;

static mpl_t* mpl_new(int nin, int* nouts, int nlayers) { 
    mpl_t* mpl = malloc(sizeof(mpl_t) + nlayers * sizeof(layer_t*));
    mpl->layers = (layer_t**)(mpl + 1);
    mpl->n = nlayers;
    mpl->nin = nin;
    mpl->nout = nouts[nlayers - 1];

    for (int i = 0; i < nlayers; i++) {
        mpl->layers[i] = layer_new(nin, nouts[i], i == (nlayers - 1) ? 0 : 1);
        nin = nouts[i];
    }   

    return mpl;
}

static void mpl_free(mpl_t* mpl) {
    if(mpl != NULL) {
        for (int i = 0; i < mpl->n; i++) {
            layer_free(mpl->layers[i]);
        }
        free(mpl);
    }
}

static value_t** mpl_forward(mpl_t* mpl, value_t** x) {  
    for (int i = 0; i < mpl->n; i++) {
        x = layer_forward(mpl->layers[i], x);
    }
    return x;
}

static void mpl_step(mpl_t* mpl, adamw_t* optimizer) {
    for (int i = 0; i < mpl->n; i++) {
        optimizer->t++;
        layer_step(mpl->layers[i], optimizer);
    }
}

static void mpl_zero_grad(mpl_t* mpl) {
    for (int i = 0; i < mpl->n; i++) {
        layer_zero_grad(mpl->layers[i]);
    }
}

static int mpl_size(mpl_t* mpl) {
    int size = 0;
    for (int i = 0; i < mpl->n; i++) {
        size += layer_size(mpl->layers[i]);
    }
    return size;
}

value_t* cross_entropy(value_t** logits, int n, int target) {
    /*subtract the max for numerical stability (avoids overflow)
    commenting these two lines out to get a cleaner visualization
    max_val = max(val.data for val in logits)
    logits = [val - max_val for val in logits]
    1) evaluate elementwise e^x
    2) compute the sum of the above
    */
    value_t* sum = value_gc_new(0, "", NULL, NULL, NULL);
    for (int i = 0; i < n; i++) {
        logits[i] = value_exp(logits[i]); //exp of logits
        sum = value_add(sum, logits[i]);  //sum of all logits
    }
    //3) normalize by the sum to get probabilities
    for (int i = 0; i < n; i++) {
       logits[i] = value_div(logits[i], sum); //divide by sum of all logits
    }
    //4) log the probabilities at target
    //5) the negative log likelihood loss (invert so we get a loss - lower is better)
    return value_neg(value_log(logits[target])); //-log(logits[target])
}

typedef struct {
    value_t* x[POS_SIZE]; //input position 
    int c;
}data_t;

typedef struct {
    data_t* data;
    int n;
    data_t* train;
    data_t* val;
    data_t* test;
    int train_size;
    int val_size;
    int test_size;
}dataset_t;

// load a dataset(train, val, test)
static dataset_t* load_dataset(int n) {
    dataset_t* dataset = malloc(sizeof(dataset_t) + n * sizeof(data_t));
    dataset->data = (data_t*)(dataset + 1);
    dataset->n = n;

    //split the dataset into train, val, test
    dataset->train = dataset->data;
    dataset->train_size = n * 8 / 10;
    dataset->val = dataset->data + dataset->train_size;
    dataset->val_size = n * 1 / 10;
    dataset->test = dataset->val + dataset->val_size;
    dataset->test_size = n * 1 / 10;

    //generate data
    yinyang_sample_t* samples = malloc(n * sizeof(yinyang_sample_t));
    yinyang_gen_data(samples, &rng, 0.5, 0.1, n);

    //transform samples to scalars
    yinyang_sample_t* sp = samples;
    data_t* dp = dataset->data;
   for (int i = 0; i < n; i++) {
        dp->x[0] = value_new(sp->pos[0]);
        dp->x[1] = value_new(sp->pos[1]);
        dp->c = sp->label;
        dp++;
        sp++;
   }

   free(samples);

    return dataset;
}

// free the dataset
static void free_dataset(dataset_t* dataset) {
    if(dataset != NULL) {
        data_t* dp = dataset->data;
        for(int i = 0; i < dataset->n; i++) {
            value_free(dp->x[0]);
            value_free(dp->x[1]);
            dp++;
        }
        free(dataset);
    }
}

// evaluate the loss function on a given data split
static value_t* loss_fun(mpl_t* mpl, data_t* data, int n) {
    value_t* loss = value_gc_new(0, "", NULL, NULL, NULL);
    for (int i = 0; i < n; i++) {
        value_t** out = mpl_forward(mpl, data[i].x);
        loss = value_add(loss, cross_entropy(out, mpl->nout, data[i].c));
    }
    //return the mean loss
    return value_div_v(loss, n);
}

void training() {
    const int step_num = 100;
    const int sample_num = 1000;
    mpl_t* mpl;
    int layers[] = {8, 3};
    adamw_t optimizer = {1, 1e-1, 0.9, 0.95, 1e-8, 1e-4};

    random_init(&rng, 42);
    mpl = mpl_new(POS_SIZE, layers, sizeof(layers) / sizeof(int));
    size_t size = POS_SIZE + mpl_size(mpl) + mpl->nout * 4 + 2;
    printf("model's value objects: %lu, value size: %lu\n", size * sample_num, sizeof(value_t));

    table_t* visited = table_new(size * sample_num, sizeof(value_t));

    //load dataset（train, val, test）
    dataset_t* data_set = load_dataset(sample_num);

    int64_t start_time = get_time_us();

    for (int i = 0; i < step_num; i++) {
        if(i % 10 == 0) {
            value_t* loss = loss_fun(mpl, data_set->val, data_set->val_size);
            printf("step %d/%d, val loss: %.6f\n", i + 1, step_num, loss->v);
            clean_gc();
        }
        value_t* loss = loss_fun(mpl, data_set->train, data_set->train_size);
        //backpropagate the loss
        backward(loss, visited);
        //update the model's weights and biases using the optimizer
        mpl_step(mpl, &optimizer);
        //zero the gradients
        mpl_zero_grad(mpl);
       //free the values of the visited table from this round of backward
        clean_gc();
        printf("step %d/%d, train loss: %.6f\n", i + 1, step_num, loss->v);
    }

    value_t* loss = loss_fun(mpl, data_set->test, data_set->test_size);
    printf("test loss: %.6f\n", loss->v);

    int64_t end_time = get_time_us();
    printf("C time taken(steps: %d, samples: %d): %.4f s\n", step_num, sample_num, (end_time - start_time)/1000000.0f);

    //free dataset
    free_dataset(data_set);

    //free mpl
    mpl_free(mpl);
    //free the fixed values required by mpl
    clean_gc();
    //free the visited table
    table_free(visited);
    //free all value objects
    free_all_slabs();
}

void test_micrograd() {
    value_t* a, *b, *h;
    table_t* visited = table_new(128, sizeof(value_t));

    a = value_new(-4.0f);
    b = value_new(2.0f);
    h = value_new(10.0f);


    value_t* c = value_add(a, b); //c = a + b
    value_t* d = value_add(value_mul(a, b), value_pow_v(b, 3)); //d = a * b + b**3

    c = value_add(c, value_add_v(c, 1)); // c+ = c + 1
    c = value_add(c, value_add(value_add_v(c, 1), value_neg(a))); //c += 1 + c - a

    d = value_add(d, value_add(value_mul_v(d, 2), value_relu(value_add(b, a)))); //d += d * 2 + (b + a).relu()
    d = value_add(d, value_add(value_mul_v(d, 3), value_relu(value_sub(b, a)))); //d += 3 * d + (b - a).relu()

    value_t* e = value_sub(c, d); //e = c - d
    value_t* f = value_pow_v(e, 2); //f = e**2
    value_t* g = value_div_v(f, 2); //g = f / 2

    g = value_add(g, value_div(h, f)); //g += 10 / f

    printf("%.6f\n", g->v); // value = 24.704082

    backward(g, visited);
    print_topo(g);

    clean_gc();
    
    printf("%.6f\n", a->grad); //value = 138.833817
    printf("%.6f\n", b->grad); //value = 645.577271

    value_free(a);
    value_free(b);
    value_free(h);

    table_free(visited);    

    free_all_slabs();
}
