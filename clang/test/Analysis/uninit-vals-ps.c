// RUN: clang -checker-simple -verify %s

struct FPRec {
  void (*my_func)(int * x);  
};

int bar(int x);

int f1_a(struct FPRec* foo) {
  int x;
  (*foo->my_func)(&x);
  return bar(x)+1; // no-warning
}

int f1_b() {
  int x;
  return bar(x)+1;  // expected-warning{{Pass-by-value argument in function is undefined.}}
}

int f2() {
  
  int x;
  
  if (x+1)  // expected-warning{{Branch}}
    return 1;
    
  return 2;  
}

int f2_b() {
  int x;
  
  return ((x+1)+2+((x))) + 1 ? 1 : 2; // expected-warning{{Branch}}
}

int f3(void) {
  int i;
  int *p = &i;
  if (*p > 0) // expected-warning{{Branch condition evaluates to an uninitialized value}}
    return 0;
  else
    return 1;
}
