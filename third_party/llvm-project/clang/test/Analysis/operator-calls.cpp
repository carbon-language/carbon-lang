// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,alpha.core,debug.ExprInspection -verify %s
void clang_analyzer_eval(bool);

struct X0 { };
bool operator==(const X0&, const X0&);

// PR7287
struct test { int a[2]; };

void t2() {
  test p = {{1,2}};
  test q;
  q = p;
}

bool PR7287(X0 a, X0 b) {
  return operator==(a, b);
}


// Inlining non-static member operators mistakenly treated 'this' as the first
// argument for a while.

struct IntComparable {
  bool operator==(int x) const {
    return x == 0;
  }
};

void testMemberOperator(IntComparable B) {
  clang_analyzer_eval(B == 0); // expected-warning{{TRUE}}
}



namespace UserDefinedConversions {
  class Convertible {
  public:
    operator int() const {
      return 42;
    }
    operator bool() const {
      return true;
    }
  };

  void test(const Convertible &obj) {
    clang_analyzer_eval((int)obj == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(obj); // expected-warning{{TRUE}}
  }
}


namespace RValues {
  struct SmallOpaque {
    float x;
    int operator +() const {
      return (int)x;
    }
  };

  struct LargeOpaque {
    float x[4];
    int operator +() const {
      return (int)x[0];
    }
  };

  SmallOpaque getSmallOpaque() {
    SmallOpaque obj;
    obj.x = 1.0;
    return obj;
  }

  LargeOpaque getLargeOpaque() {
    LargeOpaque obj = LargeOpaque();
    obj.x[0] = 1.0;
    return obj;
  }

  void test(int coin) {
    // Force a cache-out when we try to conjure a temporary region for the operator call.
    // ...then, don't crash.
    clang_analyzer_eval(+(coin ? getSmallOpaque() : getSmallOpaque())); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(+(coin ? getLargeOpaque() : getLargeOpaque())); // expected-warning{{UNKNOWN}}
  }
}

namespace SynthesizedAssignment {
  struct A {
    int a;
    A& operator=(A& other) { a = -other.a; return *this; }
    A& operator=(A&& other) { a = other.a+1; return *this; }
  };

  struct B {
    int x;
    A a[3];
    B& operator=(B&) = default;
    B& operator=(B&&) = default;
  };

  // This used to produce a warning about the iteration variable in the
  // synthesized assignment operator being undefined.
  void testNoWarning() {
    B v, u;
    u = v;
  }

  void testNoWarningMove() {
    B v, u;
    u = static_cast<B &&>(v);
  }

  void testConsistency() {
    B v, u;
    v.a[1].a = 47;
    v.a[2].a = 42;
    u = v;
    clang_analyzer_eval(u.a[1].a == -47); // expected-warning{{TRUE}}
    clang_analyzer_eval(u.a[2].a == -42); // expected-warning{{TRUE}}
  }

  void testConsistencyMove() {
    B v, u;
    v.a[1].a = 47;
    v.a[2].a = 42;
    u = static_cast<B &&>(v);
    clang_analyzer_eval(u.a[1].a == 48); // expected-warning{{TRUE}}
    clang_analyzer_eval(u.a[2].a == 43); // expected-warning{{TRUE}}
  }
}
