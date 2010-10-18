// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace Override {

namespace Test1 {

struct A {
  virtual ~A();
};

struct [[base_check]] B : A { 
  virtual ~B();
};

}

namespace Test2 {

struct A { 
  virtual void f(); // expected-note {{overridden virtual function is here}}
};

struct [[base_check]] B : A {
  virtual void f(); // expected-error {{'f' overrides function without being marked 'override'}}
};
  
}

namespace Test3 {

struct A { 
  virtual void f(); // expected-note {{overridden virtual function is here}}
};

struct B { 
  virtual void f(); // expected-note {{overridden virtual function is here}}
};

struct [[base_check]] C : A, B {
  virtual void f(); // expected-error {{'f' overrides functions without being marked 'override'}}
};
  
}

}
