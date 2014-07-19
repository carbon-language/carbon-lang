// RUN: %clang_cc1 %s -ast-print | FileCheck %s

typedef void func_typedef();
func_typedef xxx;

typedef void func_t(int x);
func_t a;

struct blah {
  struct {
    struct {
      int b;
    };
  };
};

int foo(const struct blah *b) {
  // CHECK: return b->b;
  return b->b;
}

int arr(int a[static 3]) {
  // CHECK: int a[static 3]
  return a[2];
}

int rarr(int a[restrict static 3]) {
  // CHECK: int a[restrict static 3]
  return a[2];
}

int varr(int n, int a[static n]) {
  // CHECK: int a[static n]
  return a[2];
}

int rvarr(int n, int a[restrict static n]) {
  // CHECK: int a[restrict static n]
  return a[2];
}

