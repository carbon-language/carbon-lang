// RUN: clang-cc -fsyntax-only %s

class C {
  friend class D;
};

class A {
public:
	void f();
};

class B {
  // 'A' here should refer to the declaration above.  
  friend class A;

 void f(A *a) { a->f(); }
};
