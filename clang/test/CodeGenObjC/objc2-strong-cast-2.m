// RUN: clang-cc -triple x86_64-darwin-10 -fobjc-gc -emit-llvm -o %t %s &&
// RUN: grep objc_assign_strongCast %t | count 4 &&
// RUN: true

@interface A
@end

typedef struct s0 {
  A *a[4];
} T;

T g0;

void f0(id x) {
  g0.a[0] = x;
}

void f1(id x) {
  ((T*) &g0)->a[0] = x;
}

void f2(unsigned idx)
{
   id *keys;
   keys[idx] = 0;
}

