#include <unistd.h>

volatile int release_flag = 0;

int main(int argc, char const *argv[])
{
    while (! release_flag) // Wait for debugger to attach
        sleep(3);

    return 0;
}
