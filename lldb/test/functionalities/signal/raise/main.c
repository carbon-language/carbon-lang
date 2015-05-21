#include <signal.h>
#include <stdio.h>
#include <string.h>

int main (int argc, char *argv[])
{
    if (argc < 2)
    {
        puts("Please specify a signal to raise");
        return 1;
    }

    if (strcmp(argv[1], "SIGSTOP") == 0)
        raise(SIGSTOP);
    else
    {
        printf("Unknown signal: %s\n", argv[1]);
        return 2;
    }

    return 0;
}

