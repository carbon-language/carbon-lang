/* -*- mode: c -*-
 * $Id$
 * http://www.bagley.org/~doug/shootout/
 */

int atoi(char *);
void printf(char *, unsigned long);

unsigned long
fib(unsigned long n) {
    if (n < 2)
	return(1);
    else
	return(fib(n-2) + fib(n-1));
}

int
main(int argc, char *argv[]) {
    int N = ((argc == 2) ? atoi(argv[1]) : 15);
    printf("%ld\n", fib(N));
    return(0);
}
