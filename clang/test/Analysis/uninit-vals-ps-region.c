// RUN: clang-cc -analyze -checker-simple -analyzer-store=region -verify %s

struct s {
  int data;
};

struct s global;

void g(int);

void f4() {
  int a;
  if (global.data == 0)
    a = 3;
  if (global.data == 0) // When the true branch is feasible 'a = 3'.
    g(a); // no-warning
}
