// RUN: clang -fsyntax-only -verify %s
void f(int i, int j, int k = 3);
void f(int i, int j, int k);
void f(int i, int j = 2, int k);
void f(int i, int j, int k);
void f(int i = 1, int j, int k);
void f(int i, int j, int k);

void i()
{
  f();
  f(0);
  f(0, 1);
  f(0, 1, 2);
}


int f1(int i, int i, int j) { // expected-error {{redefinition of parameter 'i'}}
  i = 17;
  return j;
} 

int x;
void g(int x, int y = x); // expected-error {{default argument references parameter 'x'}}

void h()
{
   int i;
   extern void h2(int x = sizeof(i)); // expected-error {{default argument references local variable 'i' of enclosing function}}
}

void g2(int x, int y, int z = x + y); // expected-error {{default argument references parameter 'x'}} expected-error {{default argument references parameter 'y'}}

void nondecl(int (*f)(int x = 5)) // {expected-error {{default arguments can only be specified}}}
{
  void (*f2)(int = 17)  // {expected-error {{default arguments can only be specified}}}
    = (void (*)(int = 42))f; // {expected-error {{default arguments can only be specified}}}
}
