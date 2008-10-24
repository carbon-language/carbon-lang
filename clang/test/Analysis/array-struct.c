// RUN: clang -checker-simple -verify %s

struct s {
  int data;
  int data_array[10];
};

void f(void) {
  int a[10];
  int (*p)[10];
  p = &a;
  (*p)[3] = 1;
  
  struct s d;
  struct s *q;
  q = &d;
  q->data = 3;
  d.data_array[9] = 17;
}
