/*
 * Check that we can compile helloworld
 * RUN: llvmc2 %s -o %t
 * RUN: ./%t | grep hello
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
