// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref %s -analyzer-store=basic -verify
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref %s -analyzer-store=region -verify

unsigned foo();
typedef struct bf { unsigned x:2; } bf;
void bar() {
  bf y;
  *(unsigned*)&y = foo();
  y.x = 1;
}

struct s {
  int n;
};

void f() {
  struct s a;
  int *p = &(a.n) + 1;
}
