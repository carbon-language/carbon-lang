// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Test the C++0x-specific reference initialization rules, e.g., the
// rules for rvalue references.
template<typename T> T prvalue();
template<typename T> T&& xvalue();
template<typename T> T& lvalue();

struct Base { };
struct Derived : Base { };

struct HasArray {
  int array[5];
};

int f(int);

void test_rvalue_refs() {
  // If the initializer expression...

  //   - is an xvalue, class prvalue, array prvalue or function lvalue
  //     and "cv1 T1" is reference-compatible with "cv2 T2", or

  // xvalue case
  Base&& base0 = xvalue<Base>();
  Base&& base1 = xvalue<Derived>();
  int&& int0 = xvalue<int>();

  // class prvalue case
  Base&& base2 = prvalue<Base>();
  Base&& base3 = prvalue<Derived>();

  // FIXME: array prvalue case
  //  int (&&array0)[5] = HasArray().array;

  // function lvalue case
  int (&&function0)(int) = f;
}
