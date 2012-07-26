// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-store region -cfg-add-initializers -cfg-add-implicit-dtors -verify %s

// We don't inline constructors unless we have both initializers and
// implicit destructors turned on.

void clang_analyzer_eval(bool);

class A {
  int x;
public:
  A();
};

A::A() : x(0) {
  clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
}


class DirectMember {
  int x;
public:
  DirectMember(int value) : x(value) {}

  int getX() { return x; }
};

void testDirectMember() {
  DirectMember obj(3);
  clang_analyzer_eval(obj.getX() == 3); // expected-warning{{TRUE}}
}


class IndirectMember {
  struct {
    int x;
  };
public:
  IndirectMember(int value) : x(value) {}

  int getX() { return x; }
};

void testIndirectMember() {
  IndirectMember obj(3);
  clang_analyzer_eval(obj.getX() == 3); // expected-warning{{TRUE}}
}
