/* For copyright information, see olden_v1.0/COPYRIGHT */

/**********************************************************
 * poisson.c: handles math routines for health.c          *
 **********************************************************/

#include <stdio.h>
#include <math.h>

/* From health.h */
#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

float my_rand(long idum) 
{
  long  k;
  float answer;
  
  idum ^= MASK;
  k = idum / IQ;
  idum = IA * (idum - k * IQ) - IR * k;
  idum ^= MASK;
  if (idum < 0) 
    idum  += IM;
  answer = AM * idum;
  return answer; 
}

int
main(int argc, char** argv)
{
  printf("my_rand(%d) = %g\n", 2555540, my_rand(2555540));
  printf("my_rand(%d) = %g\n", 2427763, my_rand(2427763));
  return 0;
}


