// RUN: clang-cc -fsyntax-only -faccess-control -verify %s

namespace T1 {

class A {
  virtual int f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual void f(); // expected-error{{virtual function 'f' has a different return type ('void') than the function it overrides (which has return type 'int')}}
};

}

namespace T2 {

struct a { };
struct b { };
  
class A {
  virtual a* f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual b* f(); // expected-error{{return type of virtual function 'f' is not covariant with the return type of the function it overrides ('struct T2::b *' is not derived from 'struct T2::a *')}}
};

}

namespace T3 {

struct a { };
struct b : private a { }; // expected-note{{'private' inheritance specifier here}}
  
class A {
  virtual a* f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual b* f(); // expected-error{{return type of virtual function 'f' is not covariant with the return type of the function it overrides (conversion from 'struct T3::b' to inaccessible base class 'struct T3::a')}}
};

}

namespace T4 {

struct a { };
struct a1 : a { };
struct b : a, a1 { };
  
class A {
  virtual a* f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual b* f(); // expected-error{{return type of virtual function 'f' is not covariant with the return type of the function it overrides (ambiguous conversion from derived class 'struct T4::b' to base class 'struct T4::a':\n\
    struct T4::b -> struct T4::a\n\
    struct T4::b -> struct T4::a1 -> struct T4::a)}}
};

}

namespace T5 {
  
struct a { };

class A {
  virtual a* const f(); 
  virtual a* const g(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual a* const f(); 
  virtual a* g(); // expected-error{{return type of virtual function 'g' is not covariant with the return type of the function it overrides ('struct T5::a *' has different qualifiers than 'struct T5::a *const')}}
};

}

namespace T6 {
  
struct a { };

class A {
  virtual const a* f(); 
  virtual a* g(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual a* f(); 
  virtual const a* g(); // expected-error{{return type of virtual function 'g' is not covariant with the return type of the function it overrides (class type 'struct T6::a const *' is more qualified than class type 'struct T6::a *'}}
};

}

namespace T7 {
  struct a { };
  struct b { };

  class A {
    a* f();
  };

  class B : A {
    virtual b* f();
  };
}
