// RUN: clang %s -emit-llvm
typedef struct a b;

b* x;

struct a {
  b* p;
};

void f() {
  b* z = x->p;
}
