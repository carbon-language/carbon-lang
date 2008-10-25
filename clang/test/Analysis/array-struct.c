// RUN: clang -checker-simple -verify %s
// RUN: clang -checker-simple -analyzer-store-region -verify %s

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

void f2() {
  char *p = "/usr/local";
  char (*q)[4];
  q = &"abc";
}
