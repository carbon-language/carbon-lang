#include <stdlib.h>

int printf(const char * __restrict format, ...);

typedef struct {
    int a;
    int b;
} FILE;

int main()
{
    FILE *myFile = malloc(sizeof(FILE));

    myFile->a = 5;
    myFile->b = 9;

    printf("%d\n", myFile->a + myFile->b); // Set breakpoint 0 here.

    free(myFile);
}
