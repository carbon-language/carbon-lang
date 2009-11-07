// RUN: clang -fsyntax-only -Wunused-variable -verify %s

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
