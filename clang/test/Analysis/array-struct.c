// RUN: clang -checker-simple -verify %s
// DISABLE: clang -checker-simple -analyzer-store-region -verify %s

struct s {
  int data;
  int data_array[10];
};

typedef struct {
  int data;
} STYPE;

void g1(struct s* p);

// Array to pointer conversion. Array in the struct field.
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

// StringLiteral in lvalue context and pointer to array type.
// p: ElementRegion, q: StringRegion
void f2() {
  char *p = "/usr/local";
  char (*q)[4];
  q = &"abc";
}

// Typedef'ed struct definition.
void f3() {
  STYPE s;
}

// Initialize array with InitExprList.
void f4() {
  int a[] = { 1, 2, 3};
  int b[3] = { 1, 2 };
}

// Struct variable in lvalue context.
// Assign UnknownVal to the whole struct.
void f5() {
  struct s data;
  g1(&data);
}

// AllocaRegion test.
void f6() {
  char *p;
  p = __builtin_alloca(10); 
  p[1] = 'a';
}

struct s2;

void g2(struct s2 *p);

// Incomplete struct pointer used as function argument.
void f7() {
  struct s2 *p = __builtin_alloca(10);
  g2(p);
}

// sizeof() is unsigned while -1 is signed in array index.
void f8() {
  int a[10];
  a[sizeof(a)/sizeof(int) - 1] = 1; // no-warning
}

// Initialization of struct array elements.
void f9() {
  struct s a[10];
}

// Initializing array with string literal.
void f10() {
  char a1[4] = "abc";
  char a3[6] = "abc";
}
