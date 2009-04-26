// RUN: clang-cc %s -verify -fsyntax-only -pedantic

int test1() {
  typedef int x[test1()];  // vla
  static int y = sizeof(x);  // expected-error {{not a compile-time constant}}
}

// PR2347
void f (unsigned int m)
{
  int e[2][m];

  e[0][0] = 0;
}

// PR3048
int x = sizeof(struct{char qq[x];}); // expected-error {{fields must have a constant size}}

// PR2352
void f2(unsigned int m)
{
  extern int e1[2][m]; // expected-error {{variable length array declaration can not have 'extern' linkage}}

  e1[0][0] = 0;
  
}

// PR2361
int i; 
int c[][i]; // expected-error {{variably modified type declaration not allowed at file scope}}
int d[i]; // expected-error {{variable length array declaration not allowed at file scope}}

int (*e)[i]; // expected-error {{variably modified type declaration not allowed at file scope}}

void f3()
{
  static int a[i]; // expected-error {{variable length array declaration can not have 'static' storage duration}}
  extern int b[i]; // expected-error {{variable length array declaration can not have 'extern' linkage}}

  extern int (*c1)[i]; // expected-error {{variably modified type declaration can not have 'extern' linkage}}
  static int (*d)[i];
}

// PR3663
static const unsigned array[((2 * (int)((((4) / 2) + 1.0/3.0) * (4) - 1e-8)) + 1)]; // expected-warning {{size of static array must be an integer constant expression}}

int a[*]; // expected-error {{star modifier used outside of function prototype}}
int f4(int a[*][*]);
