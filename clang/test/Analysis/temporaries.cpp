// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify -w %s

extern bool clang_analyzer_eval(bool);

struct Trivial {
  Trivial(int x) : value(x) {}
  int value;
};

struct NonTrivial : public Trivial {
  NonTrivial(int x) : Trivial(x) {}
  ~NonTrivial();
};


Trivial getTrivial() {
  return Trivial(42); // no-warning
}

const Trivial &getTrivialRef() {
  return Trivial(42); // expected-warning {{Address of stack memory associated with temporary object of type 'Trivial' returned to caller}}
}


NonTrivial getNonTrivial() {
  return NonTrivial(42); // no-warning
}

const NonTrivial &getNonTrivialRef() {
  return NonTrivial(42); // expected-warning {{Address of stack memory associated with temporary object of type 'NonTrivial' returned to caller}}
}

namespace rdar13265460 {
  struct TrivialSubclass : public Trivial {
    TrivialSubclass(int x) : Trivial(x), anotherValue(-x) {}
    int anotherValue;
  };

  TrivialSubclass getTrivialSub() {
    TrivialSubclass obj(1);
    obj.value = 42;
    obj.anotherValue = -42;
    return obj;
  }

  void testImmediate() {
    TrivialSubclass obj = getTrivialSub();

    clang_analyzer_eval(obj.value == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(obj.anotherValue == -42); // expected-warning{{TRUE}}

    clang_analyzer_eval(getTrivialSub().value == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(getTrivialSub().anotherValue == -42); // expected-warning{{TRUE}}
  }

  void testMaterializeTemporaryExpr() {
    const TrivialSubclass &ref = getTrivialSub();
    clang_analyzer_eval(ref.value == 42); // expected-warning{{TRUE}}

    const Trivial &baseRef = getTrivialSub();
    clang_analyzer_eval(baseRef.value == 42); // expected-warning{{TRUE}}
  }
}

namespace rdar13281951 {
  struct Derived : public Trivial {
    Derived(int value) : Trivial(value), value2(-value) {}
    int value2;
  };

  void test() {
    Derived obj(1);
    obj.value = 42;
    const Trivial * const &pointerRef = &obj;
    clang_analyzer_eval(pointerRef->value == 42); // expected-warning{{TRUE}}
  }
}

