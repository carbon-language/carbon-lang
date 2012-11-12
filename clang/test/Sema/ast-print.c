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
