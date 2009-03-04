// RUN: clang %s -emit-llvm -o %t -fblocks -f__block
#include <stdio.h>

int main() {
    __block int a;
    int b=2;
    a=1;
    printf("a is %d, b is %d\n", a, b);
    ^{ a = 10; printf("a is %d, b is %d\n", a, b); }();
    printf("a is %d, b is %d\n", a, b);
    a = 1;
    printf("a is %d, b is %d\n", a, b);
    return 0;
}
