/*
 * Check that the -opt switch works.
 * RUN: llvmc %s -opt -o %t
 * RUN: %abs_tmp | grep hello
 * XFAIL: vg_leak
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
