// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core %s -analyzer-store=region -verify

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

typedef struct {
  int x,y;
} Point;

Point getit(void);
void test() {
  Point p;
  (void)(p = getit()).x;
}
