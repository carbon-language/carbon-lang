/*
 * Check that we can compile helloworld
 * RUN: llvmc %s -o %t
 * RUN: %abs_tmp | grep hello
 * XFAIL: vg_leak
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
