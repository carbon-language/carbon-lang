// RUN: clang %s -fsyntax-only
// PR1892
void f(double a[restrict][5]);  // should promote to restrict ptr.
void f(double (* restrict a)[5]);

int foo (__const char *__path);
int foo(__const char *__restrict __file);

void g(int (*)(const void **, const void **));
void g(int (*compar)()) {
}

