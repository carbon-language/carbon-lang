/* Include the headers we use in int_lib.h, to verify that they work. */

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Force us to link at least one symbol in a system library
// to detect systems where we don't have those for a given
// architecture.
int main(int argc, const char **argv) {
    int x;
    memcpy(&x,&argc,sizeof(int));
}
