#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

static const char *const RETVAL_PREFIX = "retval:";
static const char *const SLEEP_PREFIX  = "sleep:";
static const char *const STDERR_PREFIX = "stderr:";

int main (int argc, char **argv)
{
    int return_value = 0;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strstr (argv[i], STDERR_PREFIX))
        {
            // Treat remainder as text to go to stderr.
            std::cerr << (argv[i] + strlen (STDERR_PREFIX)) << std::endl;
        }
        else if (std::strstr (argv[i], RETVAL_PREFIX))
        {
            // Treat as the return value for the program.
            return_value = std::atoi (argv[i] + strlen (RETVAL_PREFIX));
        }
        else if (std::strstr (argv[i], SLEEP_PREFIX))
        {
            // Treat as the amount of time to have this process sleep (in seconds).
            const int sleep_seconds = std::atoi (argv[i] + strlen (SLEEP_PREFIX));
			const int sleep_result = sleep(sleep_seconds);
			printf("sleep result: %d\n", sleep_result);
        }
        else
        {
            // Treat the argument as text for stdout.
            std::cout << argv[i] << std::endl;
        }
    }
    return return_value;
}
