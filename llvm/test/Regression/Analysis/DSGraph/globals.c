/* Test globals used and unused within different parts of a program */

#include <stdlib.h>

extern void exit_dummy(int*);

static int** G;
static int N, M;

void
foo(int *Z)          /* accesses globals printf and format string, and */
{                    /* N = alloca(int) from test() */
  if (Z == 0) exit_dummy(Z);            /* call to external function */
  ++*Z;
  printf("N = %d\n", *Z);
}

void leaf2(int* Y)
{
  if (Y == 0) exit_dummy(Y);            /* second call to external function */
}

void
test(int* X)         /* accesses global G */
{                    /* allocates G = malloc(int*) and N = alloca(int) */
  if (X == 0)
    X = &N;
  G = (int**) alloca(sizeof(int*));
  *G = &N;
  **G = 10;
  foo(*G);
  leaf2(*G);
  *X = **G;
  /* free(G); */
}

int
main()               /* only accesses global N */
{
  /* N = 0; */
  test(0 /*&N*/);
  return 0;
}
