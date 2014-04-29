#include <cstdlib>
#include <cstring>
#include <iostream>

static const char *const STDERR_PREFIX = "stderr:";
static const char *const RETVAL_PREFIX = "retval:";

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
        else
        {
            // Treat the argument as text for stdout.
            std::cout << argv[i] << std::endl;
        }
    }
    return return_value;
}
