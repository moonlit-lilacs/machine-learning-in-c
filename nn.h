#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} mat;

typedef struct {
    size_t count;
    mat *ws;
    mat *bs;
    mat *as; // activations is count+1 to account for initial data
} nn;


float rand_float(void);
float sigmoidf(float x);
#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])


#define mat_at(m, i, j) (m).es[(i)*(m).stride + (j)]
mat mat_alloc(size_t rows, size_t cols);
void mat_fill(mat m, float val);
void mat_rand(mat m, float low, float high);
mat mat_row(mat m, size_t row);
mat mat_sub(mat m, size_t n);
mat mat_col(mat m, size_t col);
void mat_copy(mat dst, mat src);
void mat_dot(mat dst, mat a, mat b);
void mat_sum(mat dst, mat a);
void mat_sig(mat m);
void mat_print(mat a, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)


nn nn_alloc(size_t *arch, size_t arch_count);
void nn_print(nn net, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(nn net, float low, float high);
void nn_forward(nn net);
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
float nn_cost(nn net, mat ti, mat to);
void nn_finite_diff(nn net, nn g, float eps, mat ti, mat to);
void nn_back_prop(nn net, nn g, mat ti, mat to);
void nn_learn(nn net, nn g, float rate);
void nn_zero(nn net);


#endif //NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}
float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}




void mat_sig(mat m){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            mat_at(m,i,j) = sigmoidf(mat_at(m,i,j));
        }
    }
}

mat mat_alloc(size_t rows, size_t cols){
    mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = calloc(rows*cols, sizeof(*m.es));
    assert(m.es != NULL);
    return m;
}

void mat_fill(mat m, float val){
    for (size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            mat_at(m,i,j) = val;
        }
    }
}

void mat_rand(mat m, float low, float high){
    for (size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            mat_at(m,i,j) = rand_float()*(high-low) + low;
        }
    }
}


void mat_dot(mat dst, mat a, mat b){
    
    assert(a.cols == b.rows);
    size_t n = a.cols;
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            mat_at(dst, i, j) = 0;
            for(size_t k = 0; k < n; k++){
                mat_at(dst, i, j) += mat_at(a,i,k) * mat_at(b,k,j);
            }
        }
    }
}
void mat_sum(mat dst, mat a){
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            mat_at(dst, i, j) += mat_at(a, i, j);
        }
    }
}

mat mat_row(mat m, size_t row){
    return (mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &mat_at(m, row, 0),
    };
}

mat mat_col(mat m, size_t col);


void mat_copy(mat dst, mat src){
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            mat_at(dst, i, j) = mat_at(src, i, j);
        }
    }
}

void mat_print(mat m, const char *name, size_t padding){
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; i++){
        printf("%*s    ", (int) padding, "");
        for(size_t j = 0; j < m.cols; j++){
            printf("%f ", mat_at(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n\n", (int) padding, "");
}

nn nn_alloc(size_t *arch, size_t arch_count){
    assert(arch_count > 0);
    
    nn net;
    net.count = arch_count-1;

    net.ws = malloc(sizeof(*net.ws)*net.count);
    assert(net.ws != NULL);
    net.bs = malloc(sizeof(*net.bs)*net.count);
    assert(net.bs != NULL);
    net.as = malloc(sizeof(*net.as)*(net.count + 1));
    assert(net.as != NULL);

    net.as[0] = mat_alloc(1, arch[0]);
    for(size_t i = 1; i < arch_count; i++){
        net.ws[i-1] = mat_alloc(net.as[i-1].cols, arch[i]);
        net.bs[i-1] = mat_alloc(1, arch[i]);
        net.as[i] = mat_alloc(1, arch[i]);
    }


    return net;
}

void nn_print(nn net, const char *name){

    char buf[256];
    printf("%s = [\n", name);

    // mat *ws = net.ws;
    // mat *bs = net.bs;

    for(size_t i = 0; i < net.count; i++){
        snprintf(buf, sizeof(buf), "ws%u", i);
        mat_print(net.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%u", i);
        mat_print(net.bs[i], buf, 4);
    }

    printf("]\n");
}

void nn_rand(nn net, float low, float high){
    for(size_t i = 0; i < net.count; i++){
        mat_rand(net.ws[i], low, high);
        mat_rand(net.bs[i], low, high);
    }
}

void nn_forward(nn net){
    for(size_t i = 0; i < net.count; i++){
        mat_dot(net.as[i+1], net.as[i], net.ws[i]);
        mat_sum(net.as[i+1], net.bs[i]);
        mat_sig(net.as[i+1]);
    }
}

float nn_cost(nn net, mat ti, mat to){
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(net).cols);
    size_t n = ti.rows;

    float c = 0;
    for(size_t i = 0; i < n; i++){
        mat x = mat_row(ti, i);
        mat y = mat_row(to, i);
        mat_copy(NN_INPUT(net), x);
        nn_forward(net);
        size_t q = to.cols;
        for(size_t j = 0; j < q; j++){
            float d = mat_at(NN_OUTPUT(net), 0, j) - mat_at(y,0,j);
            c += d*d;
        }
    }
    return c/n;
}


void nn_finite_diff(nn net, nn g, float eps, mat ti, mat to){
    float saved;
    float c = nn_cost(net,ti,to);

    for(size_t i = 0; i < net.count; i ++){
        for(size_t j = 0; j < net.ws[i].rows; j++){
            for(size_t k = 0; k < net.ws[i].cols; k++){
                saved = mat_at(net.ws[i],j,k);
                mat_at(net.ws[i],j,k) += eps;
                mat_at(g.ws[i],j,k) = (nn_cost(net, ti, to) - c)/eps;
                mat_at(net.ws[i],j,k) = saved;
            }
        }
        
        for(size_t j = 0; j < net.bs[i].rows; j++){
            for(size_t k = 0; k < net.bs[i].cols; k++){
                saved = mat_at(net.bs[i], j,k);
                mat_at(net.bs[i],j,k) += eps;
                mat_at(g.bs[i],j,k) = (nn_cost(net, ti, to) - c)/eps;
                mat_at(net.bs[i],j,k) = saved;
            }
        }
    }

}

void nn_zero(nn net){
    for(size_t i = 0; i < net.count; ++i){
        mat_fill(net.ws[i], 0);
        mat_fill(net.bs[i], 0);
        mat_fill(net.as[i], 0);
    }
    mat_fill(net.as[net.count], 0);


}

void nn_back_prop(nn net, nn g, mat ti, mat to){
    assert(ti.rows == to.rows);
    size_t n = ti.rows;
    assert(NN_OUTPUT(net).cols == to.cols);

    nn_zero(g);
    // i = current sample
    // l = current layer
    // j = current activation
    // k = previous activation

    for(size_t i = 0; i < n; ++i){
        //copying the inputs of the neural network into our Training Input (TI) helper matrix
        mat_copy(NN_INPUT(net), mat_row(ti, i));

        //pushing the input all the way through our network
        nn_forward(net);


        //this manually zeroes out the gradient activations. due to my use of `calloc` rather than `malloc`, this is technically unnecessary,
        //but is kept to maintain continuity between the presenter and myself. also, it never hurts to be safe c:
        for(size_t j = 0; j <= net.count; ++j){
            mat_fill(g.as[j], 0);
        }

        //set the output of the neural network equal to the difference between the actual output
        //and the expected output for use in our back prop.
        for(size_t j = 0; j < to.cols; ++j){
            mat_at(NN_OUTPUT(g), 0, j) = mat_at(NN_OUTPUT(net), 0,j) - mat_at(to, i, j);
        }


        //for each activation in each layer
        for(size_t l = net.count; l > 0; --l){
            for(size_t j = 0; j < net.as[l].cols; ++j){
                //j = weight matrix column
                //k  = weight matrix row


                //calculating bias:
                //a and da here only depend on the variables of the current layer because when taking the
                //partial derivative w.r.t the bias, all prior activations do not appear.
                //Note: g.as is unused before back prop so we can store activation derivatives in the gradient for later
                float a = mat_at(net.as[l], 0, j);
                float da = mat_at(g.as[l], 0, j);
                //adds the derivative calculation for the bias to the gradient for calculation later
                mat_at(g.bs[l-1],0,j) += 2*da*a*(1-a);

                //looping through the prior layer's activations
                for(size_t k = 0; k < net.as[l-1].cols; ++k){
                    //prior activation, for use in partial derivative of the weights.
                    float pa = mat_at(net.as[l-1], 0, k);

                    //prior weight, for use in partial derivative of the activations.
                    float w = mat_at(net.ws[l-1], k, j);

                    //partial derivative of the weights, added to the gradient
                    mat_at(g.ws[l-1], k, j) += 2*da*a*(1-a)*pa;

                    //partial derivative of the activations, added to the gradient
                    mat_at(g.as[l-1], 0, k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }

    for(size_t i = 0; i < g.count; ++i){
        for(size_t j = 0; j < g.ws[i].rows; ++j){
            for(size_t k = 0; k < g.ws[i].cols; ++k){
                mat_at(g.ws[i], j, k) /= n;

            }
        }
        for(size_t j = 0; j < g.bs[i].rows; ++j){
            for(size_t k = 0; k < g.bs[i].cols; ++k){
                mat_at(g.bs[i], j, k) /= n;
            }
        }
    }
}


void nn_learn(nn net, nn g, float rate){
    for(size_t i = 0; i < net.count; ++i){
        for(size_t j = 0; j < net.ws[i].rows; ++j){
            for(size_t k = 0; k < net.ws[i].cols; ++k){
                mat_at(net.ws[i],j,k) -= rate*mat_at(g.ws[i],j,k);
            }
        }
        
        for(size_t j = 0; j < net.bs[i].rows; ++j){
            for(size_t k = 0; k < net.bs[i].cols; ++k){
                mat_at(net.bs[i],j,k) -= rate*mat_at(g.bs[i],j,k);
            }
        }
    }
}





#endif //NN_IMPLEMENTATION