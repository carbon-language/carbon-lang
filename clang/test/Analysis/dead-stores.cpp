// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -warn-dead-stores -verify -Wno-unreachable-code %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -analyzer-constraints=basic -warn-dead-stores -verify -Wno-unreachable-code %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -analyzer-constraints=range -warn-dead-stores -verify -Wno-unreachable-code %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=basic -warn-dead-stores -verify -Wno-unreachable-code %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -warn-dead-stores -verify -Wno-unreachable-code %s

//===----------------------------------------------------------------------===//
// Basic dead store checking (but in C++ mode).
//===----------------------------------------------------------------------===//

int j;
void test1() {
  int x = 4;

  ++x; // expected-warning{{never read}}

  switch (j) {
  case 1:
    throw 1;
    (void)x;
    break;
  }
}

//===----------------------------------------------------------------------===//
// Dead store checking involving constructors.
//===----------------------------------------------------------------------===//

class Test2 {
  int &x;
public:
  Test2(int &y) : x(y) {}
  ~Test2() { ++x; }
};

int test2(int x) {
  { Test2 a(x); } // no-warning
  return x;
}

//===----------------------------------------------------------------------===//
// Dead store checking involving CXXTemporaryExprs
//===----------------------------------------------------------------------===//

namespace TestTemp {
  template<typename _Tp>
  class pencil {
  public:
    ~pencil() throw() {}
  };
  template<typename _Tp, typename _Number2> struct _Row_base {
    _Row_base(const pencil<_Tp>& x) {}
  };
  template<typename _Tp, typename _Number2 = TestTemp::pencil<_Tp> >
  class row : protected _Row_base<_Tp, _Number2>     {
    typedef _Row_base<_Tp, _Number2> _Base;
    typedef _Number2 pencil_type;
  public:
    explicit row(const pencil_type& __a = pencil_type()) : _Base(__a) {}
  };
}

void test2_b() {
  TestTemp::row<const char*> x; // no-warning
}

//===----------------------------------------------------------------------===//
// Test references.
//===----------------------------------------------------------------------===//

void test3_a(int x) {
  ++x; // expected-warning{{never read}}
}

void test3_b(int &x) {
  ++x; // no-warninge
}

void test3_c(int x) {
  int &y = x;
  // Shows the limitation of dead stores tracking.  The write is really
  // dead since the value cannot escape the function.
  ++y; // no-warning
}

void test3_d(int &x) {
  int &y = x;
  ++y; // no-warning
}

void test3_e(int &x) {
  int &y = x;
}

