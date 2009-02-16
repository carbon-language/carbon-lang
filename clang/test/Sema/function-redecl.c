// RUN: clang -fsyntax-only -verify %s

// PR3588
void g0(int, int);
void g0(); // expected-note{{previous declaration is here}}

void f0() {
  g0(1, 2, 3); // expected-error{{too many arguments to function call}}
}

void g0(int); // expected-error{{conflicting types for 'g0'}}

int g1(int, int);

typedef int INT;

INT g1(x, y)
     int x;
     int y;
{
  return x + y;
}

int g2(int, int); // expected-note{{previous declaration is here}}

INT g2(x) // expected-error{{conflicting types for 'g2'}}
     int x;
{
  return x;
}
