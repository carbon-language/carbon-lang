// RUN: clang -checker-simple -verify %s

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

void* compound_literal(int x) {
  if (x)
    return &(unsigned short){((unsigned short)0x22EF)}; // expected-warning{{Address of stack memory}} expected-warning{{braces around scalar initializer}}

  int* array[] = {};
  struct s { int z; double y; int w; };
  return &((struct s){ 2, 0.4, 5 * 8 }); // expected-warning{{Address of stack memory}}
}

