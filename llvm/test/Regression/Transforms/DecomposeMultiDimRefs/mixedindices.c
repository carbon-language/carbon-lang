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
AddMixed(Mixed_t* M)
{
  double sum = 0;
  int i, j;
  
  for (i=0; i < 10; ++i)
    sum += M->A[i];
  
  for (i=0; i < 10; ++i)
    for (j=0; j < 10; ++j)
      sum += M->B[i][j];
  
  for (i=0; i < 10; ++i) {
    sum += (double) M->F[i].c;
    sum += M->F[i].x;
  }
  
  return sum;
}

void
InitializeMixed(Mixed_t* M, int base)
{
  int i, j;
  
  for (i=0; i < 10; ++i)
    M->A[i] = i + base;
    
  for (i=0; i < 10; ++i)
    for (j=0; j < 10; ++j)
      M->B[i][j] = i*10 + j + base;
  
  for (i=0; i < 10; ++i) {
    M->F[i].c = 'Q';
    M->F[i].x = i / 10 + base;
  }
}

int
main(int argc, char** argv)
{
  Mixed_t M;
  Mixed_t MA[4];
  int i;
  
  InitializeMixed(&M, 100);
  printf("Sum(M)  = %.2f\n", AddMixed(&M));
  
  for (i=0; i < 4; i++) {
    InitializeMixed(&MA[i], 100 * (i+2));
    printf("Sum(MA[%d]) = %.2f\n", i, AddMixed(&MA[i]));
  }

  return 0;
}
