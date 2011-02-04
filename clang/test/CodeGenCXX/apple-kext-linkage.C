// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -emit-llvm -o - %s | FileCheck %s

struct Base {
  virtual ~Base();
} ;

struct Derived : Base {
  void operator delete(void *) { }
  Derived();
};

void foo() {
  Derived d1;			// ok
}

inline unsigned f(unsigned n) { return n == 0 ? 0 : n + f(n-1); }

unsigned g(unsigned n) { return f(n); }


template <typename X> X ident(X x) { return x; }
int foo(int n) { return ident(n); }

// CHECK-NOT: define linkonce_odr
// CHECK 5 : define internal
