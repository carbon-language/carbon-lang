// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -fobjc-arc -analyzer-config c++-inlining=constructors -Wno-null-dereference -std=c++11 -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

// A simplified version of std::move.
template <typename T>
T &&move(T &obj) {
  return static_cast<T &&>(obj);
}


struct Wrapper {
  __strong id obj;
};

void test() {
  Wrapper w;
  // force a diagnostic
  *(char *)0 = 1; // expected-warning{{Dereference of null pointer}}
}


struct IntWrapper {
  int x;
};

void testCopyConstructor() {
  IntWrapper a;
  a.x = 42;

  IntWrapper b(a);
  clang_analyzer_eval(b.x == 42); // expected-warning{{TRUE}}
}

struct NonPODIntWrapper {
  int x;

  virtual int get();
};

void testNonPODCopyConstructor() {
  NonPODIntWrapper a;
  a.x = 42;

  NonPODIntWrapper b(a);
  clang_analyzer_eval(b.x == 42); // expected-warning{{TRUE}}
}


namespace ConstructorVirtualCalls {
  class A {
  public:
    int *out1, *out2, *out3;

    virtual int get() { return 1; }

    A(int *out1) {
      *out1 = get();
    }
  };

  class B : public A {
  public:
    virtual int get() { return 2; }

    B(int *out1, int *out2) : A(out1) {
      *out2 = get();
    }
  };

  class C : public B {
  public:
    virtual int get() { return 3; }

    C(int *out1, int *out2, int *out3) : B(out1, out2) {
      *out3 = get();
    }
  };

  void test() {
    int a, b, c;

    C obj(&a, &b, &c);
    clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}

    clang_analyzer_eval(obj.get() == 3); // expected-warning{{TRUE}}

    // Sanity check for devirtualization.
    A *base = &obj;
    clang_analyzer_eval(base->get() == 3); // expected-warning{{TRUE}}
  }
}

namespace TemporaryConstructor {
  class BoolWrapper {
  public:
    BoolWrapper() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      value = true;
    }
    bool value;
  };

  void test() {
    // PR13717 - Don't crash when a CXXTemporaryObjectExpr is inlined.
    if (BoolWrapper().value)
      return;
  }
}


namespace ConstructorUsedAsRValue {
  using TemporaryConstructor::BoolWrapper;

  bool extractValue(BoolWrapper b) {
    return b.value;
  }

  void test() {
    bool result = extractValue(BoolWrapper());
    clang_analyzer_eval(result); // expected-warning{{TRUE}}
  }
}

namespace PODUninitialized {
  class POD {
  public:
    int x, y;
  };

  class PODWrapper {
  public:
    POD p;
  };

  class NonPOD {
  public:
    int x, y;

    NonPOD() {}
    NonPOD(const NonPOD &Other)
      : x(Other.x), y(Other.y) // expected-warning {{undefined}}
    {
    }
    NonPOD(NonPOD &&Other)
    : x(Other.x), y(Other.y) // expected-warning {{undefined}}
    {
    }

    NonPOD &operator=(const NonPOD &Other)
    {
      x = Other.x;
      y = Other.y; // expected-warning {{undefined}}
      return *this;
    }
    NonPOD &operator=(NonPOD &&Other)
    {
      x = Other.x;
      y = Other.y; // expected-warning {{undefined}}
      return *this;
    }
  };

  class NonPODWrapper {
  public:
    class Inner {
    public:
      int x, y;

      Inner() {}
      Inner(const Inner &Other)
        : x(Other.x), y(Other.y) // expected-warning {{undefined}}
      {
      }
      Inner(Inner &&Other)
      : x(Other.x), y(Other.y) // expected-warning {{undefined}}
      {
      }

      Inner &operator=(const Inner &Other)
      {
        x = Other.x; // expected-warning {{undefined}}
        y = Other.y;
        return *this;
      }
      Inner &operator=(Inner &&Other)
      {
        x = Other.x; // expected-warning {{undefined}}
        y = Other.y;
        return *this;
      }
    };

    Inner p;
  };

  void testPOD() {
    POD p;
    p.x = 1;
    POD p2 = p; // no-warning
    clang_analyzer_eval(p2.x == 1); // expected-warning{{TRUE}}
    POD p3 = move(p); // no-warning
    clang_analyzer_eval(p3.x == 1); // expected-warning{{TRUE}}

    // Use rvalues as well.
    clang_analyzer_eval(POD(p3).x == 1); // expected-warning{{TRUE}}

    PODWrapper w;
    w.p.y = 1;
    PODWrapper w2 = w; // no-warning
    clang_analyzer_eval(w2.p.y == 1); // expected-warning{{TRUE}}
    PODWrapper w3 = move(w); // no-warning
    clang_analyzer_eval(w3.p.y == 1); // expected-warning{{TRUE}}

    // Use rvalues as well.
    clang_analyzer_eval(PODWrapper(w3).p.y == 1); // expected-warning{{TRUE}}
  }

  void testNonPOD() {
    NonPOD p;
    p.x = 1;
    NonPOD p2 = p;
  }

  void testNonPODMove() {
    NonPOD p;
    p.x = 1;
    NonPOD p2 = move(p);
  }

  void testNonPODWrapper() {
    NonPODWrapper w;
    w.p.y = 1;
    NonPODWrapper w2 = w;
  }

  void testNonPODWrapperMove() {
    NonPODWrapper w;
    w.p.y = 1;
    NonPODWrapper w2 = move(w);
  }

  // Not strictly about constructors, but trivial assignment operators should
  // essentially work the same way.
  namespace AssignmentOperator {
    void testPOD() {
      POD p;
      p.x = 1;
      POD p2;
      p2 = p; // no-warning
      clang_analyzer_eval(p2.x == 1); // expected-warning{{TRUE}}
      POD p3;
      p3 = move(p); // no-warning
      clang_analyzer_eval(p3.x == 1); // expected-warning{{TRUE}}

      PODWrapper w;
      w.p.y = 1;
      PODWrapper w2;
      w2 = w; // no-warning
      clang_analyzer_eval(w2.p.y == 1); // expected-warning{{TRUE}}
      PODWrapper w3;
      w3 = move(w); // no-warning
      clang_analyzer_eval(w3.p.y == 1); // expected-warning{{TRUE}}
    }

    void testReturnValue() {
      POD p;
      p.x = 1;
      POD p2;
      clang_analyzer_eval(&(p2 = p) == &p2); // expected-warning{{TRUE}}

      PODWrapper w;
      w.p.y = 1;
      PODWrapper w2;
      clang_analyzer_eval(&(w2 = w) == &w2); // expected-warning{{TRUE}}
    }

    void testNonPOD() {
      NonPOD p;
      p.x = 1;
      NonPOD p2;
      p2 = p;
    }

    void testNonPODMove() {
      NonPOD p;
      p.x = 1;
      NonPOD p2;
      p2 = move(p);
    }

    void testNonPODWrapper() {
      NonPODWrapper w;
      w.p.y = 1;
      NonPODWrapper w2;
      w2 = w;
    }

    void testNonPODWrapperMove() {
      NonPODWrapper w;
      w.p.y = 1;
      NonPODWrapper w2;
      w2 = move(w);
    }
  }
}

namespace ArrayMembers {
  struct Primitive {
    int values[3];
  };

  void testPrimitive() {
    Primitive a = { { 1, 2, 3 } };

    clang_analyzer_eval(a.values[0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2] == 3); // expected-warning{{TRUE}}

    Primitive b = a;

    clang_analyzer_eval(b.values[0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[2] == 3); // expected-warning{{TRUE}}

    Primitive c;
    c = b;

    clang_analyzer_eval(c.values[0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[2] == 3); // expected-warning{{TRUE}}
  }

  struct NestedPrimitive {
    int values[2][3];
  };

  void testNestedPrimitive() {
    NestedPrimitive a = { { { 0, 0, 0 }, { 1, 2, 3 } } };

    clang_analyzer_eval(a.values[1][0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][2] == 3); // expected-warning{{TRUE}}

    NestedPrimitive b = a;

    clang_analyzer_eval(b.values[1][0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][2] == 3); // expected-warning{{TRUE}}

    NestedPrimitive c;
    c = b;

    clang_analyzer_eval(c.values[1][0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][2] == 3); // expected-warning{{TRUE}}
  }

  struct POD {
    IntWrapper values[3];
  };

  void testPOD() {
    POD a = { { { 1 }, { 2 }, { 3 } } };

    clang_analyzer_eval(a.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2].x == 3); // expected-warning{{TRUE}}

    POD b = a;

    clang_analyzer_eval(b.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[2].x == 3); // expected-warning{{TRUE}}

    POD c;
    c = b;

    clang_analyzer_eval(c.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[2].x == 3); // expected-warning{{TRUE}}
  }

  struct NestedPOD {
    IntWrapper values[2][3];
  };

  void testNestedPOD() {
    NestedPOD a = { { { { 0 }, { 0 }, { 0 } }, { { 1 }, { 2 }, { 3 } } } };

    clang_analyzer_eval(a.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][2].x == 3); // expected-warning{{TRUE}}

    NestedPOD b = a;

    clang_analyzer_eval(b.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][2].x == 3); // expected-warning{{TRUE}}

    NestedPOD c;
    c = b;

    clang_analyzer_eval(c.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][2].x == 3); // expected-warning{{TRUE}}
  }

  struct NonPOD {
    NonPODIntWrapper values[3];
  };

  void testNonPOD() {
    NonPOD a;
    a.values[0].x = 1;
    a.values[1].x = 2;
    a.values[2].x = 3;

    clang_analyzer_eval(a.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2].x == 3); // expected-warning{{TRUE}}

    NonPOD b = a;

    clang_analyzer_eval(b.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[2].x == 3); // expected-warning{{UNKNOWN}}

    NonPOD c;
    c = b;

    clang_analyzer_eval(c.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[2].x == 3); // expected-warning{{UNKNOWN}}
  }

  struct NestedNonPOD {
    NonPODIntWrapper values[2][3];
  };

  void testNestedNonPOD() {
    NestedNonPOD a;
    a.values[0][0].x = 0;
    a.values[0][1].x = 0;
    a.values[0][2].x = 0;
    a.values[1][0].x = 1;
    a.values[1][1].x = 2;
    a.values[1][2].x = 3;

    clang_analyzer_eval(a.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][2].x == 3); // expected-warning{{TRUE}}

    NestedNonPOD b = a;

    clang_analyzer_eval(b.values[1][0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1][1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1][2].x == 3); // expected-warning{{UNKNOWN}}

    NestedNonPOD c;
    c = b;

    clang_analyzer_eval(c.values[1][0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1][1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1][2].x == 3); // expected-warning{{UNKNOWN}}
  }
  
  struct NonPODDefaulted {
    NonPODIntWrapper values[3];

    NonPODDefaulted() = default;
    NonPODDefaulted(const NonPODDefaulted &) = default;
    NonPODDefaulted &operator=(const NonPODDefaulted &) = default;
  };

  void testNonPODDefaulted() {
    NonPODDefaulted a;
    a.values[0].x = 1;
    a.values[1].x = 2;
    a.values[2].x = 3;

    clang_analyzer_eval(a.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2].x == 3); // expected-warning{{TRUE}}

    NonPODDefaulted b = a;

    clang_analyzer_eval(b.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[2].x == 3); // expected-warning{{UNKNOWN}}

    NonPODDefaulted c;
    c = b;

    clang_analyzer_eval(c.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[2].x == 3); // expected-warning{{UNKNOWN}}
  }
};
