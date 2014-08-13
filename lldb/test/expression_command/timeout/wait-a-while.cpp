#include <stdio.h>
#include <stdint.h>

#include <chrono>
#include <thread>


int
wait_a_while (int microseconds)
{
    int num_times = 0;

    while (1)
    {
        num_times++;
        std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
        break;
    }
    return num_times;
}

int
main (int argc, char **argv)
{
    printf ("stop here in main.\n");
    int num_times = wait_a_while (argc * 1000);
    printf ("Done, took %d times.\n", num_times);

    return 0;

}
