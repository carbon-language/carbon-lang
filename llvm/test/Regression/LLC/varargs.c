#include <stdio.h>
#include <stdlib.h>

#define A	16807.0
#define M	2147483647.0

/*
 * This function calls floor() which does not have a prototype.
 * Test that the argument to floor is passed correctly.
 */
double
my_rand(double seed)
{
    double t = A*seed  + 1; 
    double floor();

    seed = t - (M * floor(t / M));      /* t%M if t > M; t otherwise */
    return seed;

} /* end of random */


int
main(int argc, char** argv)
{
  double seed = 123 * ((argc > 1)? atof(argv[1]) : 3.1415926);
  printf("my_rand(%lf) = %lf\n", seed, my_rand(seed));
  return 0;
}
