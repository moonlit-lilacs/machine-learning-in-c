#define NN_IMPLEMENTATION
#include <stdio.h>
#include <time.h>
#include "nn.h"


#define BITS 3

int main(){

    srand(69);
    const float rate = 1;

    size_t n = (1  << BITS);
    size_t rows = n*n;

    mat ti = mat_alloc(rows, 2*BITS);
    mat to = mat_alloc(rows, BITS+1);


    for(size_t i = 0; i < ti.rows; ++i){
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        printf("%u, %u, %u\n", x, y, z);
        for(size_t j = 0; j < BITS; ++j){
            mat_at(ti, i, j)        = (x >> j)&1;
            mat_at(ti, i, j + BITS) = (y >> j)&1;
            mat_at(to, i, j)          = (z >> j)&1;
        }
        mat_at(to, i, BITS) = z >= n;
    }

    //MAT_PRINT(ti);
    //MAT_PRINT(to);
    
    // for(size_t hidden = 0; hidden <= 10; hidden += 2){
        // size_t arch[] = {2*BITS, hidden, BITS+1};
    
        // nn net = nn_alloc(arch, ARRAY_LEN(arch));
        // nn g = nn_alloc(arch, ARRAY_LEN(arch));

        // nn_rand(net, 0, 1);
        // //NN_PRINT(net);

        // for(size_t i = 0; i < 5*1000; ++i){
        //         nn_back_prop(net, g, ti, to);
        //         nn_learn(net, g, rate);
        // }
    //     printf("%u hidden neurons: cost = %f\n", hidden, nn_cost(net,ti,to));
    // }

    size_t arch[] = {2*BITS, 3*BITS, BITS+1};
        
    nn net = nn_alloc(arch, ARRAY_LEN(arch));
    nn g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(net, 0, 1);
    //NN_PRINT(net);

    printf("cost = %f\n", nn_cost(net, ti, to));
    for(size_t i = 0; i < 10*1000; ++i){
            nn_back_prop(net, g, ti, to);
            nn_learn(net, g, rate);
            printf("%u: cost = %f\n", i, nn_cost(net, ti, to));
    }
    printf("cost = %f\n", nn_cost(net, ti, to));

    size_t fails = 0;

    for(size_t x = 0; x < n; ++x){
        for(size_t y = 0; y < n; ++y){
            size_t z = x +y;
            
            for(size_t j = 0; j < BITS; ++j){
                mat_at(NN_INPUT(net), 0, j)        = (x >> j)&1;
                mat_at(NN_INPUT(net), 0, j + BITS) = (y >> j)&1;
            }
            nn_forward(net);
            if((mat_at(NN_OUTPUT(net), 0, BITS) > 0.5f)){
                if (z < n){
                    printf("%u + %u = (OVERFLOW <> %u)\n", x, y, z);
                    fails += 1;
                }
            }
            else{
                size_t a = 0;
                for(size_t j = 0; j < BITS; ++j){
                    size_t bit = mat_at(NN_OUTPUT(net), 0, j) > 0.5f;
                    a |= bit << j;
                }
                if(z != a){
                    printf("%u + %u = (%u <> %u)\n", x, y, z, a);
                }
            }

            

        }
    }

    if (fails == 0) {printf("OK\n");}


//     size_t arch[] = {2*BITS, 6, BITS+1};
    
//     nn net = nn_alloc(arch, ARRAY_LEN(arch));
//     nn g = nn_alloc(arch, ARRAY_LEN(arch));

//     nn_rand(net, 0, 1);
//     NN_PRINT(net);

//     printf("cost = %f\n", nn_cost(net, ti, to));

//    for(size_t i = 0; i < 10000; ++i){
//         nn_back_prop(net, g, ti, to);
//         nn_learn(net, g, rate);
//         printf("%u: cost = %f\n", i, nn_cost(net,ti,to));
//         fflush(stdout); 
//    }
    






    return 0;   

}