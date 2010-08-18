// RUN: %clang_cc1 -fsyntax-only -verify %s
class X { 
public:
  operator bool();
  operator int() const;

  bool f() {
    return operator bool();
  }

  float g() {
    return operator float(); // expected-error{{use of undeclared 'operator float'}}
  }
};

operator int(); // expected-error{{conversion function must be a non-static member function}}

operator int; // expected-error{{'operator int' cannot be the name of a variable or data member}}

typedef int func_type(int);
typedef int array_type[10];

class Y {
public:
  void operator bool(int, ...) const; // expected-error{{conversion function cannot have a return type}} \
  // expected-error{{conversion function cannot have any parameters}}
  
  operator float(...) const;  // expected-error{{conversion function cannot be variadic}}
  
  
  operator func_type(); // expected-error{{conversion function cannot convert to a function type}}
  operator array_type(); // expected-error{{conversion function cannot convert to an array type}}
};


typedef int INT;
typedef INT* INT_PTR;

class Z { 
  operator int(); // expected-note {{previous declaration is here}}
  operator int**(); // expected-note {{previous declaration is here}}
  
  operator INT();  // expected-error{{conversion function cannot be redeclared}}
  operator INT_PTR*(); // expected-error{{conversion function cannot be redeclared}}
};


class A { };

class B : public A {
public:
  operator A&() const; // expected-warning{{conversion function converting 'B' to its base class 'A' will never be used}}
  operator const void() const; // expected-warning{{conversion function converting 'B' to 'void const' will never be used}}
  operator const B(); // expected-warning{{conversion function converting 'B' to itself will never be used}}
};

// This used to crash Clang.
struct Flip;
struct Flop {
  Flop();
  Flop(const Flip&); // expected-note{{candidate constructor}}
};
struct Flip {
  operator Flop() const; // expected-note{{candidate function}}
};
Flop flop = Flip(); // expected-error {{conversion from 'Flip' to 'Flop' is ambiguous}}

// This tests that we don't add the second conversion declaration to the list of user conversions
struct C {
  operator const char *() const;
};

C::operator const char*() const { return 0; }

void f(const C& c) {
  const char* v = c;
}

// Test. Conversion in base class is visible in derived class.
class XB { 
public:
  operator int(); // expected-note {{candidate function}}
};

class Yb : public XB { 
public:
  operator char(); // expected-note {{candidate function}}
};

void f(Yb& a) {
  if (a) { } // expected-error {{conversion from 'Yb' to 'bool' is ambiguous}}
  int i = a; // OK. calls XB::operator int();
  char ch = a;  // OK. calls Yb::operator char();
}

// Test conversion + copy construction.
class AutoPtrRef { };

class AutoPtr {
  AutoPtr(AutoPtr &); // expected-note{{declared private here}}
  
public:
  AutoPtr();
  AutoPtr(AutoPtrRef);
  
  operator AutoPtrRef();
};

AutoPtr make_auto_ptr();

AutoPtr test_auto_ptr(bool Cond) {
  AutoPtr p1( make_auto_ptr() );
  
  AutoPtr p;
  if (Cond)
    return p; // expected-error{{calling a private constructor}}
  
  return AutoPtr();
}

struct A1 {
  A1(const char *);
  ~A1();

private:
  A1(const A1&); // expected-note 2 {{declared private here}}
};

A1 f() {
  // FIXME: redundant diagnostics!
  return "Hello"; // expected-error {{calling a private constructor}} expected-warning {{an accessible copy constructor}}
}

namespace source_locations {
  template<typename T>
  struct sneaky_int {
    typedef int type;
  };

  template<typename T, typename U>
  struct A { };

  template<typename T>
  struct A<T, T> : A<T, int> { };

  struct E {
    template<typename T>
    operator A<T, typename sneaky_int<T>::type>&() const; // expected-note{{candidate function}}
  };

  void f() {
    A<float, float> &af = E(); // expected-error{{no viable conversion}}
    A<float, int> &af2 = E();
    const A<float, int> &caf2 = E();
  }

  // Check 
  template<typename T>
  struct E2 {
    operator T
    * // expected-error{{pointer to a reference}}
    () const;
  };

  E2<int&> e2i; // expected-note{{in instantiation}}
}

namespace crazy_declarators {
  struct A {
    (&operator bool())(); // expected-error {{must use a typedef to declare a conversion to 'bool (&)()'}}

    // FIXME: This diagnostic is misleading (the correct spelling
    // would be 'operator int*'), but it's a corner case of a
    // rarely-used syntax extension.
    *operator int();  // expected-error {{must use a typedef to declare a conversion to 'int *'}}
  };
}

namespace smart_ptr {
  class Y { 
    class YRef { };

    Y(Y&);

  public:
    Y();
    Y(YRef);

    operator YRef(); // expected-note{{candidate function}}
  };

  struct X { // expected-note{{candidate constructor (the implicit copy constructor) not}}
    explicit X(Y);
  };

  Y make_Y();

  X f() {
    X x = make_Y(); // expected-error{{no viable conversion from 'smart_ptr::Y' to 'smart_ptr::X'}}
    X x2(make_Y());
    return X(Y());
  }
}

struct Any {
  Any(...);
};

struct Other {
  Other(const Other &); 
  Other();
};

void test_any() {
  Any any = Other(); // expected-error{{cannot pass object of non-POD type 'Other' through variadic constructor; call will abort at runtime}}
}

namespace PR7055 {
  // Make sure that we don't allow too many conversions in an
  // auto_ptr-like template. In particular, we can't create multiple
  // temporary objects when binding to a reference.
  struct auto_ptr {
    struct auto_ptr_ref { };

    auto_ptr(auto_ptr&);
    auto_ptr(auto_ptr_ref);
    explicit auto_ptr(int *);

    operator auto_ptr_ref();
  };

  struct X {
    X(auto_ptr);
  };

  X f() {
    X x(auto_ptr(new int));
    return X(auto_ptr(new int));
  }

  auto_ptr foo();

  X e(foo());

  struct Y {
    Y(X);
  };
  
  Y f2(foo());
}

namespace PR7934 {
  typedef unsigned char uint8;

  struct MutablePtr {
    MutablePtr() : ptr(0) {}
    void *ptr;

    operator void*() { return ptr; }

  private:
    operator uint8*() { return reinterpret_cast<uint8*>(ptr); }
    operator const char*() const { return reinterpret_cast<const char*>(ptr); }
  };

  void fake_memcpy(const void *);

  void use() {
    MutablePtr ptr;
    fake_memcpy(ptr);
  }
}
