// RUN: clang %s -fsyntax-only -verify
// PR1892
void f(double a[restrict][5]);  // should promote to restrict ptr.
void f(double (* restrict a)[5]);

int foo (__const char *__path);
int foo(__const char *__restrict __file);

void func(const char*); //expected-error{{previous declaration is here}}
void func(char*); //expected-error{{conflicting types for 'func'}}

void g(int (*)(const void **, const void **));
void g(int (*compar)()) {
}

