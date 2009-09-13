/*
 * Check that the 'sink' options work.
 * RUN: llvmc -v -Wall %s -o %t |& grep "Wall"
 * RUN: %abs_tmp | grep hello
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}
