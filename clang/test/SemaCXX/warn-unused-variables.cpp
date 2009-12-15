// RUN: %clang -fsyntax-only -Wunused-variable -verify %s

template<typename T> void f() {
	T t;
	t = 17;
}

// PR5407
struct A { A(); };
struct B { ~B(); };
void f() {
  A a;
  B b;
}

// PR5531
namespace PR5531 {
  struct A {
  };

  struct B {
    B(int);
  };

  struct C {
    ~C();
  };

  void test() {
    A();
    B(17);
    C();
  }
}
