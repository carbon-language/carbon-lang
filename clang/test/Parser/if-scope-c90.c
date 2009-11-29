// RUN: clang-cc -fsyntax-only -verify -std=c90 %s

int f (int z)
{ 
  if (z > (int) sizeof (enum {a, b}))
      return a;
   return b;
} 
