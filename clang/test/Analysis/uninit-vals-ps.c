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
