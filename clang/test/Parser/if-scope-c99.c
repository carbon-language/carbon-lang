// RUN: clang-cc -fsyntax-only -verify --std=c99 %s

int f (int z)
{ 
   if (z > sizeof (enum {a, b}))
      return a;
   return b; // expected-error{{use of undeclared identifier}}
}
