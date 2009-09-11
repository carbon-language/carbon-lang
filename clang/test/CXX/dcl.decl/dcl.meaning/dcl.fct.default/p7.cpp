// RUN: clang-cc -fsyntax-only -verify %s

void h()
{
  int i;
  extern void h2(int x = sizeof(i)); // expected-error {{default argument references local variable 'i' of enclosing function}}
}
