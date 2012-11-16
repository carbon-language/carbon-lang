// RUN: %clang -ccc-cxx -fsanitize=vptr %s -O3 -o %t
// RUN: %t rT && %t mT && %t fT
// RUN: %t rU && %t mU && %t fU
// RUN: %t rS 2>&1 | FileCheck %s --check-prefix=CHECK-REFERENCE
// RUN: %t mS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER
// RUN: %t fS 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN
// RUN: %t rV 2>&1 | FileCheck %s --check-prefix=CHECK-REFERENCE
// RUN: %t mV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER
// RUN: %t fV 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN

// FIXME: This test produces linker errors on Darwin.
// XFAIL: darwin

struct S {
  S() : a(0) {}
  ~S() {}
  int a;
  int f() { return 0; }
  virtual int v() { return 0; }
};

struct T : S {
  T() : b(0) {}
  int b;
  int g() { return 0; }
  virtual int v() { return 1; }
};

struct U : S, T { virtual int v() { return 2; } };

int main(int, char **argv) {
  T t;
  (void)t.a;
  (void)t.b;
  (void)t.f();
  (void)t.g();
  (void)t.v();
  (void)t.S::v();

  U u;
  (void)u.T::a;
  (void)u.b;
  (void)u.T::f();
  (void)u.g();
  (void)u.v();
  (void)u.T::v();
  (void)((T&)u).S::v();

  T *p = 0;
  switch (argv[1][1]) {
  case 'S':
    p = reinterpret_cast<T*>(new S);
    break;
  case 'T':
    p = new T;
    break;
  case 'U':
    p = new U;
    break;
  case 'V':
    p = reinterpret_cast<T*>(new U);
    break;
  }

  switch (argv[1][0]) {
  case 'r':
    // CHECK-REFERENCE: vptr.cpp:[[@LINE+1]]:13: fatal error: reference binding to address 0x{{[0-9a-f]*}} which does not point to an object of type 'T'
    {T &r = *p;}
    break;
  case 'm':
    // CHECK-MEMBER: vptr.cpp:[[@LINE+1]]:15: fatal error: member access within address 0x{{[0-9a-f]*}} which does not point to an object of type 'T'
    return p->b;
  case 'f':
    // CHECK-MEMFUN: vptr.cpp:[[@LINE+1]]:12: fatal error: member call on address 0x{{[0-9a-f]*}} which does not point to an object of type 'T'
    return p->g();
  }
}
