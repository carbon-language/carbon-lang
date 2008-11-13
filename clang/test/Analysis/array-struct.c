// RUN: clang -checker-simple -verify %s
// DISABLED: clang -checker-simple -analyzer-store-region -verify %s

struct s {
  int data;
  int data_array[10];
};

typedef struct {
  int data;
} STYPE;

void g1(struct s* p);

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

void f3() {
  STYPE s;
}

void f4() {
  int a[] = { 1, 2, 3};
  int b[3] = { 1, 2 };
}

void f5() {
  struct s data;
  g1(&data);
}

void f6() {
  char *p;
  p = __builtin_alloca(10); 
  p[1] = 'a';
}

struct s2;

void g2(struct s2 *p);

void f7() {
  struct s2 *p = __builtin_alloca(10);
  g2(p);
}

void f8() {
  int a[10];
  a[sizeof(a) - 1] = 1;
}
