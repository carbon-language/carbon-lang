// RUN: not %clang_cc1 -O1 %s -emit-llvm
// PR6913

#include <stdio.h>

int main()
{
   int x[10][10];
   int (*p)[] = x; // expected-error {{invalid use of array with unspecified bounds}

   int i;

   for(i = 0; i < 10; ++i)
   {
       p[i][i] = i;
   }
}

// rdar://13705391
void foo(int a[*][2]) {(void)a[0][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo1(int a[2][*]) {(void)a[0][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo2(int a[*][*]) {(void)a[0][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo3(int a[2][*][2]) {(void)a[0][1][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo4(int a[2][*][*]) {(void)a[0][1][1]; } // expected-error {{variable length array must be bound in function definition}}
