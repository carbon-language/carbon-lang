// RUN: %llvmgcc %s -o /dev/null -S

#include <complex.h>

int foo(complex float c) {
    return creal(c);
}
