/*===- test/Regression/Transforms/Scalar/DecomposeMultiDimRefs.cpp     -----=*
 * 
 * This is a regression test for the DecomposeArrayRefs pass.
 * It tests several different combinations of structure and
 * array indexes in individual references.  It does *not* yet
 * sufficiently test type-unsafe operations like treating a
 * 1-D array as a 2-D or 3-D array.  (multidim.ll in this directory
 * tests a simple case of that though.)
 *===---------------------------------------------------------------------===*/

#include <stdlib.h>
#include <stdio.h>

typedef struct Flat_struct {
  char   c;
  float  x;
} Flat_t;

typedef struct Mixed_struct {
  int    N;
  double A[10];
  double B[10][10];
  Flat_t F[10];
} Mixed_t;


double
InitializeMixed(Mixed_t* M, int base)
{
  double sum = 0;
  int i, j;
  
  for (i=0; i < 10; ++i) {
    int coord;
    coord = i + base;
    M->A[i] = coord;
    sum += coord;
  }
  
  for (i=0; i < 10; ++i)
    for (j=0; j < 10; ++j) {
      int coord;
      coord = i*10 + j + base;
      M->B[i][j] = coord;
      sum += coord;
    }
  
  for (i=0; i < 10; ++i) {
    double ratio;
    M->F[i].c = 'Q';
    ratio = i / 10 + base;
    M->F[i].x = ratio;
    sum += ratio;
  }
  
  return sum;
}

int
main(int argc, char** argv)
{
  Mixed_t M;
  Mixed_t MA[4];
  int i;
  
  printf("Sum(M)  = %.2f\n", InitializeMixed(&M, 100));
  
  for (i=0; i < 4; i++)
    printf("Sum(MA[%d]) = %.2f\n", i, InitializeMixed(&MA[i], 400));

  return 0;
}
