// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5787
class C {
 public:
  ~C() {}
};

template <typename T>
class E {
 public:
  E& Foo(const C&);
  E& Bar() { return Foo(C()); }
};

void Test() {
  E<int> e;
  e.Bar();
}
