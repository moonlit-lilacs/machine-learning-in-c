#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define eps 1e-3
#define rate 1e-0

typedef struct {
    //layer 1
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;
    
    //layer 2
    float and_w1;
    float and_w2;
    float and_b;
} xor;

typedef float sample[3];

//XOR-Gate
sample xor_train[] = {
    {0,0,0},
    {1,0,1},
    {1,1,0},
    {0,1,1},
};

//OR-Gate
sample or_train[] = {
    {0,0,0},
    {1,0,1},
    {1,1,1},
    {0,1,1},
};

//AND-Gate
sample and_train[] = {
    {0,0,0},
    {1,0,0},
    {1,1,1},
    {0,1,0},
};

//NOR-Gate
sample nor_train[] = {
    {0,0,1},
    {1,0,0},
    {1,1,0},
    {0,1,0},
};

//NAND-Gate
sample nand_train[] = {
    {0,0,1},
    {1,0,1},
    {1,1,0},
    {0,1,1},
};

sample *train_data = xor_train;
size_t train_count = 4;

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x){
    return 1.f/(1.f + expf(-x));
}


float forward(xor m, float x1, float x2){
    float a = sigmoidf(x1 * m.or_w1 + x2 * m.or_w2 + m.or_b);
    float b = sigmoidf(x1 * m.nand_w1 + x2 * m.nand_w2 + m.nand_b);
    return sigmoidf(a * m.and_w1 + b * m.and_w2 + m.and_b);
}

float cost(xor m){
    float result = 0.0f;
        for(size_t i = 0; i < train_count; i++){
            float x1 = train_data[i][0];
            float x2 = train_data[i][1];
            float y = forward(m, x1, x2);
            float d = y - train_data[i][2];
            result += d*d;

        }
        return result /= train_count;
}

xor rand_xor(void){
    xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    return m;
}

void print_xor(xor m){
    printf("or w1 : %f\n", m.or_w1);
    printf("or w2: %f\n", m.or_w2);
    printf("or b : %f\n", m.or_b);
    printf("nand w1 : %f\n", m.nand_w1);
    printf("nand w2: %f\n", m.nand_w2);
    printf("nand b : %f\n", m.nand_b);
    printf("and w1 : %f\n", m.and_w1);
    printf("and w2 : %f\n", m.and_w2);
    printf("and b : %f\n", m.and_b);
}

xor finite_diff(xor m){
    float c = cost(m);
    xor g;
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c)/eps;
    m.or_w2 = saved;
    
    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c)/eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c)/eps;
    m.nand_w2 = saved;
    
    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c)/eps;
    m.nand_b = saved;
    


    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c)/eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c)/eps;
    m.and_w2 = saved;
    
    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c)/eps;
    m.and_b = saved;
    
    return g;
}

xor learn(xor m, xor g){
    m.or_w1 -= rate*g.or_w1;
    m.or_w2 -= rate*g.or_w2;
    m.or_b -= rate*g.or_b;

    m.nand_w1 -= rate*g.nand_w1;
    m.nand_w2 -= rate*g.nand_w2;
    m.nand_b -= rate*g.nand_b;
    
    //layer 2
    m.and_w1 -= rate*g.and_w1;
    m.and_w2 -= rate*g.and_w2;
    m.and_b -= rate*g.and_b;
    
    return m;
}


int main(void){
    srand(time(0));
    xor m = rand_xor();
    
    // print_xor(m);
    // printf("-----------------------------------\n");
    // print_xor(g);

    for(size_t i = 0; i < 100*1000; i++){
        xor g = finite_diff(m);
        m = learn(m,g);
        //printf("cost: %f\n", cost(m));
    }
    
    printf("\n---------------------------------------------------------\n\n");

    printf("\"or\" gate\n");
     for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            float a = sigmoidf(i * m.or_w1 + j * m.or_w2 + m.or_b);
            
            printf("%u | %u = %f\n", i, j, a);
        }
    }
    printf("\"and\" gate\n");
     for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            float b = sigmoidf(i * m.and_w1 + j * m.and_w2 + m.and_b);
            printf("%u | %u = %f\n", i, j, b);
        }
    }


    return 0;
}