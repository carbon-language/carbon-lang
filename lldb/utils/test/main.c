#include <stdio.h>

int main(int argc, const char* argv[])
{
    int *null_ptr = 0;
    printf("Hello, fault!\n");
    printf("Now segfault %d\n", *null_ptr);
}
