// RUN: %llvmgcc -O3 -xc %s -S -o - | not grep builtin

#include <math.h>

void zsqrtxxx(float num) {
   num = sqrt(num);
}

