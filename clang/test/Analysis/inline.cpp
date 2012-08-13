// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=inlining -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

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
