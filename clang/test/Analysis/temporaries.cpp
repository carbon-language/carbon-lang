// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify -w -std=c++03 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify -w -std=c++11 %s

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

namespace compound_literals {
  struct POD {
    int x, y;
  };
  struct HasCtor {
    HasCtor(int x, int y) : x(x), y(y) {}
    int x, y;
  };
  struct HasDtor {
    int x, y;
    ~HasDtor();
  };
  struct HasCtorDtor {
    HasCtorDtor(int x, int y) : x(x), y(y) {}
    ~HasCtorDtor();
    int x, y;
  };

  void test() {
    clang_analyzer_eval(((POD){1, 42}).y == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(((HasDtor){1, 42}).y == 42); // expected-warning{{TRUE}}

#if __cplusplus >= 201103L
    clang_analyzer_eval(((HasCtor){1, 42}).y == 42); // expected-warning{{TRUE}}

    // FIXME: should be TRUE, but we don't inline the constructors of
    // temporaries because we can't model their destructors yet.
    clang_analyzer_eval(((HasCtorDtor){1, 42}).y == 42); // expected-warning{{UNKNOWN}}
#endif
  }
}

