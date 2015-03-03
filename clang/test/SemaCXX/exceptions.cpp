// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

struct A; // expected-note 4 {{forward declaration of 'A'}}

struct Abstract { virtual void f() = 0; }; // expected-note {{unimplemented pure virtual method 'f'}}

void trys() {
  try {
  } catch(int i) { // expected-note {{previous definition}}
    int j = i;
    int i; // expected-error {{redefinition of 'i'}}
  } catch(float i) {
  } catch(void v) { // expected-error {{cannot catch incomplete type 'void'}}
  } catch(A a) { // expected-error {{cannot catch incomplete type 'A'}}
  } catch(A *a) { // expected-error {{cannot catch pointer to incomplete type 'A'}}
  } catch(A &a) { // expected-error {{cannot catch reference to incomplete type 'A'}}
  } catch(Abstract) { // expected-error {{variable type 'Abstract' is an abstract class}}
  } catch(...) {
    int j = i; // expected-error {{use of undeclared identifier 'i'}}
  }

  try {
  } catch(...) { // expected-error {{catch-all handler must come last}}
  } catch(int) {
  }
}

void throws() {
  throw;
  throw 0;
  throw throw; // expected-error {{cannot throw object of incomplete type 'void'}}
  throw (A*)0; // expected-error {{cannot throw pointer to object of incomplete type 'A'}}
}

void jumps() {
l1:
  goto l5;
  goto l4; // expected-error {{cannot jump}}
  goto l3; // expected-error {{cannot jump}}
  goto l2; // expected-error {{cannot jump}}
  goto l1;
  try { // expected-note 4 {{jump bypasses initialization of try block}}
  l2:
    goto l5;
    goto l4; // expected-error {{cannot jump}}
    goto l3; // expected-error {{cannot jump}}
    goto l2;
    goto l1;
  } catch(int) { // expected-note 4 {{jump bypasses initialization of catch block}}
  l3:
    goto l5;
    goto l4; // expected-error {{cannot jump}}
    goto l3;
    goto l2; // expected-error {{cannot jump}}
    goto l1;
  } catch(...) { // expected-note 4 {{jump bypasses initialization of catch block}}
  l4:
    goto l5;
    goto l4;
    goto l3; // expected-error {{cannot jump}}
    goto l2; // expected-error {{cannot jump}}
    goto l1;
  }
l5:
  goto l5;
  goto l4; // expected-error {{cannot jump}}
  goto l3; // expected-error {{cannot jump}}
  goto l2; // expected-error {{cannot jump}}
  goto l1;
}

struct BadReturn {
  BadReturn() try {
  } catch(...) {
    // Try to hide
    try {
    } catch(...) {
      {
        if (0)
          return; // expected-error {{return in the catch of a function try block of a constructor is illegal}}
      }
    }
  }
  BadReturn(int);
};

BadReturn::BadReturn(int) try {
} catch(...) {
  // Try to hide
  try {
  } catch(int) {
    return; // expected-error {{return in the catch of a function try block of a constructor is illegal}}
  } catch(...) {
    {
      if (0)
        return; // expected-error {{return in the catch of a function try block of a constructor is illegal}}
    }
  }
}

// Cannot throw an abstract type.
class foo {
public:
  foo() {}
  void bar () {
    throw *this; // expected-error{{cannot throw an object of abstract type 'foo'}}
  }
  virtual void test () = 0; // expected-note{{unimplemented pure virtual method 'test'}}
};

namespace PR6831 {
  namespace NA { struct S; }
  namespace NB { struct S; }
  
  void f() {
    using namespace NA;
    using namespace NB;
    try {
    } catch (int S) { 
    }
  }
}

namespace Decay {
  struct A {
    void f() throw (A[10]);
  };

  template<typename T> struct B {
    void f() throw (B[10]);
  };
  template struct B<int>;

  void f() throw (int[10], int(*)());
  void f() throw (int*, int());

  template<typename T> struct C {
    void f() throw (T); // expected-error {{pointer to incomplete type 'Decay::E' is not allowed in exception specification}}
  };
  struct D {
    C<D[10]> c;
  };
  struct E; // expected-note {{forward declaration}}
  C<E[10]> e; // expected-note {{in instantiation of}}
}

void rval_ref() throw (int &&); // expected-error {{rvalue reference type 'int &&' is not allowed in exception specification}} expected-warning {{C++11}}

namespace ConstVolatileThrow {
struct S {
  S() {}         // expected-note{{candidate constructor not viable}}
  S(const S &s); // expected-note{{candidate constructor not viable}}
};

typedef const volatile S CVS;

void f() {
  throw CVS(); // expected-error{{no matching constructor for initialization}}
}
}

namespace ConstVolatileCatch {
struct S {
  S() {}
  S(const volatile S &s);

private:
  S(const S &s); // expected-note {{declared private here}}
};

void f();

void g() {
  try {
    f();
  } catch (volatile S s) { // expected-error {{calling a private constructor}}
  }
}
}
