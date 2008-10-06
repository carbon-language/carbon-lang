// RUN: clang -fsyntax-only %s 

int x(1);

void f() {
  int x(1);
  for (int x(1);;) {}
}
