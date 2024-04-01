#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>


float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

float td_sum[] = {
    0,0, 0,0, 0,0,
    0,0, 0,1, 0,1,
    0,1, 0,1, 1,0,
    0,1, 1,0, 1,1,
};

int main(void){
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td+2,
    };

    
    float rate = 1;

    size_t arch[] = {2,2,1};
    nn net = nn_alloc(arch, ARRAY_LEN(arch));
    nn g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(net, 0, 1);

    
    // NN_PRINT(net);
    
    
    mat_copy(NN_INPUT(net), mat_row(ti, 1));
    nn_forward(net);


    MAT_PRINT(NN_OUTPUT(net));

    printf("cost = %f\n", nn_cost(net, ti, to));
    
    for(size_t i = 0; i < 5000; i++){
        nn_back_prop(net, g, ti, to);
        //NN_PRINT(g);
        nn_learn(net,g,rate);
        //printf("%u: cost = %f\n", i, nn_cost(net, ti, to));
    };
    printf("cost = %f\n", nn_cost(net, ti, to));
    

    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            mat_at(NN_INPUT(net), 0,0) = i;
            mat_at(NN_INPUT(net), 0,1) = j;
            nn_forward(net);
            printf("%u ^ %u = %f\n", i, j, mat_at(NN_OUTPUT(net),0,0));
        }
    }


    return 0;
}