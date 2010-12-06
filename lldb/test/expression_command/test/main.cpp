#include <stdio.h>

int main (int argc, char const *argv[])
{
    printf ("Hello world!\n");
    puts ("hello");
    // Please test many expressions while stopped at this line:
#if 0
    expr 'a'        // make sure character constant makes it down (this is broken by the command line parser code right now)
    expr 2          // Test int
    expr 2ull       // Test unsigned long long
    expr 2.234f     // Test float constants
    expr 2.234      // Test double constants
    expr 2+3
    expr argc
    expr argc + 22
    expr argv
    expr argv[0]
    expr argv[1]
    expr argv[-1]
    expr puts("bonjour")                        // Test constant strings...
    expr printf("\t\x68\n")       // Test constant strings that contain the \xXX (TAB, 'h', '\n' should be printed)
    expr printf("\"\n")       // Test constant strings that contains an escaped double quote char
    expr printf("\'\n")       // Test constant strings that contains an escaped single quote char
    expr printf ("one: %i\n", 1)
    expr printf ("1.234 as float: %f\n", 1.234f)
    expr printf ("1.234 as double: %g\n", 1.234)
    expr printf ("one: %i, two: %llu\n", 1, 2ull)
    expr printf ("two: %llu, one: %i\n", 2ull, 1)
    expr random() % 255l
#endif
    return 0;
}
