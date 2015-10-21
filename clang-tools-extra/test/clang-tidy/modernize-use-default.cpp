// RUN: %python %S/check_clang_tidy.py %s modernize-use-default %t

class A {
public:
  A();
  ~A();
};

A::A() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use '= default' to define a trivial default constructor [modernize-use-default]
// CHECK-FIXES: A::A() = default;
A::~A() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use '= default' to define a trivial destructor [modernize-use-default]
// CHECK-FIXES: A::~A() = default;

// Inline definitions.
class B {
public:
  B() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: B() = default;
  ~B() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~B() = default;
};

void f();

class C {
public:
  // Non-empty constructor body.
  C() { f(); }
  // Non-empty destructor body.
  ~C() { f(); }
};

class D {
public:
  // Constructor with initializer.
  D() : Field(5) {}
  // Constructor with arguments.
  D(int Arg1, int Arg2) {}
  int Field;
};

// Private constructor/destructor.
class E {
  E() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: E() = default;
  ~E() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~E() = default;
};

// struct.
struct F {
  F() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: F() = default;
  ~F() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: F() = default;
};

// Deleted constructor/destructor.
class G {
public:
  G() = delete;
  ~G() = delete;
};

// Do not remove other keywords.
class H {
public:
  explicit H() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: explicit H() = default;
  virtual ~H() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: virtual ~H() = default;
};

// Nested class.
struct I {
  struct II {
    II() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default'
    // CHECK-FIXES: II() = default;
    ~II() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use '= default'
    // CHECK-FIXES: ~II() = default;
  };
  int Int;
};

// Class template.
template <class T>
class J {
public:
  J() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: J() = default;
  ~J() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~J() = default;
};

// Non user-provided constructor/destructor.
struct K {
  int Int;
};
void g() {
  K *PtrK = new K();
  PtrK->~K();
  delete PtrK;
}

// Already using default.
struct L {
  L() = default;
  ~L() = default;
};
struct M {
  M();
  ~M();
};
M::M() = default;
M::~M() = default;

// Delegating constructor and overriden destructor.
struct N : H {
  N() : H() {}
  ~N() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~N() override = default;
};
