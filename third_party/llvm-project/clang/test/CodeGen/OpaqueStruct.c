// RUN: %clang_cc1 %s -emit-llvm -o %t
typedef struct a b;

b* x;

struct a {
  b* p;
};

void f() {
  b* z = x->p;
}
