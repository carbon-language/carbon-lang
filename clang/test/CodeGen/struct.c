// RUN: clang %s -emit-llvm

struct  {
  int x;
  int y;
} point;

void fn1() {
  point.x = 42;
}

/* Nested member */
struct  {
  struct {
    int a;
    int b;
  } p1;
} point2;

void fn2() {
  point2.p1.a = 42;
}

/* Indirect reference */
typedef struct __sf {
 unsigned char *c;
 short flags;
} F;

typedef struct __sf2 {
  F *ff;
} F2;

int fn3(F2 *c) {
  if (c->ff->c >= 0)
    return 1;
  else
    return 0;
}
