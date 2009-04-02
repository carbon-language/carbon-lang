// RUN: clang-cc -fsyntax-only -verify %s

int f3(y, x, 
       x)          // expected-error {{redefinition of parameter}}
  int y, x, 
      x;           // expected-error {{redefinition of parameter}}
{
  return x + y; 
} 

void f4(void) { 
  f3 (1, 1, 2, 3, 4); // expected-warning{{too many arguments}}
}

