// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-ipa=inlining -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);

// This is the standard placement new.
inline void* operator new(size_t, void* __p) throw()
{
  return __p;
}


class A {
public:
  int getZero() { return 0; }
  virtual int getNum() { return 0; }
};

void test(A &a) {
  clang_analyzer_eval(a.getZero() == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.getNum() == 0); // expected-warning{{UNKNOWN}}

  A copy(a);
  clang_analyzer_eval(copy.getZero() == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.getNum() == 0); // expected-warning{{TRUE}}
}


class One : public A {
public:
  virtual int getNum() { return 1; }
};

void testPathSensitivity(int x) {
  A a;
  One b;

  A *ptr;
  switch (x) {
  case 0:
    ptr = &a;
    break;
  case 1:
    ptr = &b;
    break;
  default:
    return;
  }

  // This should be true on both branches.
  clang_analyzer_eval(ptr->getNum() == x); // expected-warning {{TRUE}}
}


namespace PureVirtualParent {
  class Parent {
  public:
    virtual int pureVirtual() const = 0;
    int callVirtual() const {
      return pureVirtual();
    }
  };

  class Child : public Parent {
  public:
    virtual int pureVirtual() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return 42;
    }
  };

  void testVirtual() {
    Child x;

    clang_analyzer_eval(x.pureVirtual() == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(x.callVirtual() == 42); // expected-warning{{TRUE}}
  }
}


namespace PR13569 {
  class Parent {
  protected:
    int m_parent;
    virtual int impl() const = 0;

    Parent() : m_parent(0) {}

  public:
    int interface() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return impl();
    }
  };

  class Child : public Parent {
  protected:
    virtual int impl() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return m_parent + m_child;
    }

  public:
    Child() : m_child(0) {}

    int m_child;
  };

  void testVirtual() {
    Child x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }


  class Grandchild : public Child {};

  void testDevirtualizeToMiddle() {
    Grandchild x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }
}

namespace PR13569_virtual {
  class Parent {
  protected:
    int m_parent;
    virtual int impl() const = 0;

    Parent() : m_parent(0) {}

  public:
    int interface() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return impl();
    }
  };

  class Child : virtual public Parent {
  protected:
    virtual int impl() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return m_parent + m_child;
    }

  public:
    Child() : m_child(0) {}

    int m_child;
  };

  void testVirtual() {
    Child x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }


  class Grandchild : virtual public Child {};

  void testDevirtualizeToMiddle() {
    Grandchild x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }
}

namespace Invalidation {
  struct X {
    void touch(int &x) const {
      x = 0;
    }

    void touch2(int &x) const;

    virtual void touchV(int &x) const {
      x = 0;
    }

    virtual void touchV2(int &x) const;

    int test() const {
      // We were accidentally not invalidating under -analyzer-ipa=inlining
      // at one point for virtual methods with visible definitions.
      int a, b, c, d;
      touch(a);
      touch2(b);
      touchV(c);
      touchV2(d);
      return a + b + c + d; // no-warning
    }
  };
}

namespace DefaultArgs {
  int takesDefaultArgs(int i = 42) {
    return -i;
  }

  void testFunction() {
    clang_analyzer_eval(takesDefaultArgs(1) == -1); // expected-warning{{TRUE}}
    clang_analyzer_eval(takesDefaultArgs() == -42); // expected-warning{{TRUE}}
  }

  class Secret {
  public:
    static const int value = 42;
    int get(int i = value) {
      return i;
    }
  };

  void testMethod() {
    Secret obj;
    clang_analyzer_eval(obj.get(1) == 1); // expected-warning{{TRUE}}

    // FIXME: Should be 'TRUE'. See PR13673 or <rdar://problem/11720796>.
    clang_analyzer_eval(obj.get() == 42); // expected-warning{{UNKNOWN}}

    // FIXME: Even if we constrain the variable, we still have a problem.
    // See PR13385 or <rdar://problem/12156507>.
    if (Secret::value != 42)
      return;
    clang_analyzer_eval(Secret::value == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(obj.get() == 42); // expected-warning{{UNKNOWN}}
  }
}

namespace OperatorNew {
  class IntWrapper {
  public:
    int value;

    IntWrapper(int input) : value(input) {
      // We don't want this constructor to be inlined unless we can actually
      // use the proper region for operator new.
      // See PR12014 and <rdar://problem/12180598>.
      clang_analyzer_checkInlined(false); // no-warning
    }
  };

  void test() {
    IntWrapper *obj = new IntWrapper(42);
    // should be TRUE
    clang_analyzer_eval(obj->value == 42); // expected-warning{{UNKNOWN}}
  }

  void testPlacement() {
    IntWrapper *obj = static_cast<IntWrapper *>(malloc(sizeof(IntWrapper)));
    IntWrapper *alias = new (obj) IntWrapper(42);

    clang_analyzer_eval(alias == obj); // expected-warning{{TRUE}}

    // should be TRUE
    clang_analyzer_eval(obj->value == 42); // expected-warning{{UNKNOWN}}
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
