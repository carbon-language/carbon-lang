// %llvmgcc %s -S -o -

#include <math.h>
#define I 1.0iF

double __complex test(double X) { return -(X*I); }
