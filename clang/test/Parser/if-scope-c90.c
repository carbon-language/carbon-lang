// RUN: clang-cc -fsyntax-only -verify --std=c90 %s

int f (int z)
{ 
   if (z > sizeof (enum {a, b}))
      return a;
   return b;
} 
