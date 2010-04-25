// RUN: not %llvmgcc -O1 %s -S |& grep {error: invalid use of array with unspecified bounds}
// PR6913

#include <stdio.h>

int main()
{
   int x[10][10];
   int (*p)[] = x;   // <-- this line is what triggered it

   int i;

   for(i = 0; i < 10; ++i)
   {
       p[i][i] = i;
   }
}
