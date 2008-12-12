// RUN: clang %s -fsyntax-only -verify -pedantic -fpascal-strings

int a() {int p; *(1 ? &p : (void*)(0 && (a(),1))) = 10;}

// rdar://6091492 - ?: with __builtin_constant_p as the operand is an i-c-e.
int expr;
char w[__builtin_constant_p(expr) ? expr : 1];


// __builtin_constant_p as the condition of ?: allows arbitrary foldable
// constants to be transmogrified into i-c-e's.
char b[__builtin_constant_p((int)(1.0+2.0)) ? (int)(1.0+2.0) : -1];

struct c {
  int a : (  // expected-error {{expression is not an integer constant expression}}
           __builtin_constant_p((int)(1.0+2.0)) ? (int)(1.0+
     expr  // expected-note {{subexpression not valid in an integer constant expression}}
           ) : -1);
};




void test1(int n, int* p) { *(n ? p : (void *)(7-7)) = 1; }
void test2(int n, int* p) { *(n ? p : (void *)0) = 1; }



char array[1024/(sizeof (long))];

int x['\xBb' == (char) 187 ? 1: -1];

// PR1992
void func(int x)
{
  switch (x) {
    case sizeof("abc"): break;
    case sizeof("loooong"): func(4);
    case sizeof("\ploooong"): func(4);
  }
}


// rdar://4213768
int expr;
char y[__builtin_constant_p(expr) ? -1 : 1];
char z[__builtin_constant_p(4) ? 1 : -1];

