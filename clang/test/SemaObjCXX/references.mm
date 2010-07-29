// RUN: %clang_cc1 -verify -emit-llvm -o - %s

// Test reference binding.

typedef struct {
  int f0;
  int f1;
} T;

@interface A
@property (assign) T p0;
@property (assign) T& p1; // expected-error {{property of reference type is not supported}}
@end

int f0(const T& t) {
  return t.f0;
}

int f1(A *a) {
  return f0(a.p0);
}

int f2(A *a) {
  return f0(a.p1);	// expected-error {{property 'p1' not found on object of type 'A *'}}
}

// PR7740
@class NSString;

void f3(id);
void f4(NSString &tmpstr) {
  f3(&tmpstr);
}
