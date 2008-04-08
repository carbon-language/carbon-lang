// RUN: clang -fsyntax-only -verify %s
void f(int i, int j, int k = 3);
void f(int i, int j = 2, int k);
void f(int i = 1, int j, int k);

void i()
{
  f();
  f(0);
  f(0, 1);
  f(0, 1, 2);
}
