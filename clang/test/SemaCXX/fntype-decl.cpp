// RUN: clang -fsyntax-only -verify %s

// PR2942
typedef void fn(int);
fn f;

int g(int x, int y);
int g(int x, int y = 2);

typedef int g_type(int, int);
g_type g;

int h(int x) {
  return g(x);
}
