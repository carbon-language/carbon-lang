struct A {
  virtual void f(int);
};

struct B {
  virtual void f(int);
  virtual void g();
};

struct C : B, A { 
  virtual void g();
};

struct D : C {
  virtual void f(int);
};

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: overrides.cpp:11:16: CXXMethod=g:11:16 (virtual) [Overrides @7:16] Extent=[11:3 - 11:19]
// CHECK: overrides.cpp:15:16: CXXMethod=f:15:16 (virtual) [Overrides @2:16, @6:16] Extent=[15:3 - 15:22]
