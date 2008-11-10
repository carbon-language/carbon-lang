// RUN: clang -fsyntax-only -verify %s
class A { 
  int m;
};

class B : public A { 
public:
  B() : A(), m(1), n(3.14) { }

private:
  int m;
  float n;  
};


class C : public virtual B { 
public:
  C() : B() { }
};

class D : public C { 
public:
  D() : B(), C() { }
};

class E : public D, public B { 
public:
  E() : B(), D() { } // expected-error{{base class initializer 'B' names both a direct base class and an inherited virtual base class}}
};


typedef int INT;

class F : public B { 
public:
  int B;

  F() : B(17),
        m(17), // expected-error{{member initializer 'm' does not name a non-static data member or base class}}
        INT(17) // expected-error{{constructor initializer 'INT' does not name a class}}
  { 
  }
};

class G : A {
  G() : A(10); // expected-error{{expected '{'}}
};
