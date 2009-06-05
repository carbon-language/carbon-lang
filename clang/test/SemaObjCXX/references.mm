// RUN: clang-cc -verify -emit-llvm -o %t %s
// XFAIL

// Test reference binding.

typedef struct {
  int f0;
  int f1;
} T;

@interface A
@property (assign) T p0;
@property (assign) T& p1;
@end

int f0(const T& t) {
  return t.f0;
}

int f1(A *a) {
  return f0(a.p0);
}

int f2(A *a) {
  return f0(a.p1);
}

