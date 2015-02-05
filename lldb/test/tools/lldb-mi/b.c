#include <stdio.h>

extern int a_MyFunction();
int
b_MyFunction ()
{
  (void)a_MyFunction(); //BP_b_MyFunction
  printf ("b is about to return 20.\n");
  return 20;
}
