// RUN: clang-cc -analyze -checker-cfref -analyzer-store=basic -verify %s
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s

#include <stdlib.h>

int* f1() {
  int x = 0;
  return &x; // expected-warning{{Address of stack memory associated with local variable 'x' returned.}} expected-warning{{address of stack memory associated with local variable 'x' returned}}
}

int* f2(int y) {
  return &y;  // expected-warning{{Address of stack memory associated with local variable 'y' returned.}} expected-warning{{address of stack memory associated with local variable 'y' returned}}
}

int* f3(int x, int *y) {
  int w = 0;
  
  if (x)
    y = &w;
    
  return y; // expected-warning{{Address of stack memory associated with local variable 'w' returned.}}
}

void* compound_literal(int x, int y) {
  if (x)
    return &(unsigned short){((unsigned short)0x22EF)}; // expected-warning{{Address of stack memory}}

  int* array[] = {};
  struct s { int z; double y; int w; };
  
  if (y)
    return &((struct s){ 2, 0.4, 5 * 8 }); // expected-warning{{Address of stack memory}}
    
  
  void* p = &((struct s){ 42, 0.4, x ? 42 : 0 });
  return p; // expected-warning{{Address of stack memory}}
}

void* alloca_test() {
  void* p = __builtin_alloca(10);
  return p; // expected-warning{{Address of stack memory}}
}

int array_test(int x[2]) {
  return x[0]; // no-warning
}

struct baz {
  int x;
  int y[2];
};

int struct_test(struct baz byVal, int flag) {
  if (flag)  
    return byVal.x; // no-warning
  else {
    return byVal.y[0]; // no-warning
  }
}
