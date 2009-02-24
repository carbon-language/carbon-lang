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

void test() {
  int f1;
  {
    void f1(double);
    {
      void f1(double); // expected-note{{previous declaration is here}}
      {
        int f1(int); // expected-error{{conflicting types for 'f1'}}
      }
    }
  }
}

extern void g3(int); // expected-note{{previous declaration is here}}
static void g3(int x) { } // expected-error{{static declaration of 'g3' follows non-static declaration}}

void test2() {
  extern int f2; // expected-note{{previous definition is here}}
  {
    void f2(int); // expected-error{{redefinition of 'f2' as different kind of symbol}}
  }

  {
    int f2;
    {
      void f2(int); // okay
    }
  }
}
