// RUN: clang-cc -fsyntax-only -verify %s 
int f(double);
int f(int);

int (*pfd)(double) = f; // selects f(double)
int (*pfd2)(double) = &f; // selects f(double)
int (*pfd3)(double) = ((&((f)))); // selects f(double)
int (*pfi)(int) = &f;    // selects f(int)
// FIXME: This error message is not very good. We need to keep better
// track of what went wrong when the implicit conversion failed to
// give a better error message here.
int (*pfe)(...) = &f;    // expected-error{{incompatible type initializing '<overloaded function type>', expected 'int (*)(...)'}}
int (&rfi)(int) = f;     // selects f(int)
int (&rfd)(double) = f;  // selects f(double)

void g(int (*fp)(int));   // expected-note{{note: candidate function}}
void g(int (*fp)(float));
void g(int (*fp)(double)); // expected-note{{note: candidate function}}

int g1(int);
int g1(char);

int g2(int);
int g2(double);

void g_test() {
  g(g1);
  g(g2); // expected-error{{call to 'g' is ambiguous; candidates are:}}
}
