// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

typedef struct Opaque *Data;
struct IntWrapper {
  int x;
};

struct Child : public IntWrapper {
  void set() { x = 42; }
};

void test(Data data) {
  Child *wrapper = reinterpret_cast<Child*>(data);
  // Don't crash when upcasting here.
  // We don't actually know if 'data' is a Child.
  wrapper->set();
  clang_analyzer_eval(wrapper->x == 42); // expected-warning{{TRUE}}
}

namespace PR14872 {
  class Base1 {};
  class Derived1 : public Base1 {};

  Derived1 *f1();

  class Base2 {};
  class Derived2 : public Base2 {};

  void f2(Base2 *foo);

  void f3(void** out)
  {
    Base1 *v;
    v = f1();
    *out = v;
  }

  void test()
  {
    Derived2 *p;
    f3(reinterpret_cast<void**>(&p));
    // Don't crash when upcasting here.
    // In this case, 'p' actually refers to a Derived1.
    f2(p);
  }
}
