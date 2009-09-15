// RUN: clang-cc -fsyntax-only -verify %s 
class X { 
public:
  operator bool();
  operator int() const;

  bool f() {
    return operator bool();
  }

  float g() {
    return operator float(); // expected-error{{no matching function for call to 'operator float'}}
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
  operator A&() const; // expected-warning{{conversion function converting 'class B' to its base class 'class A' will never be used}}
  operator const void() const; // expected-warning{{conversion function converting 'class B' to 'void const' will never be used}}
  operator const B(); // expected-warning{{conversion function converting 'class B' to itself will never be used}}
};

// This used to crash Clang.
struct Flip;
struct Flop {
  Flop();
  Flop(const Flip&);
};
struct Flip {
  operator Flop() const;
};
Flop flop = Flip(); // expected-error {{cannot initialize 'flop' with an rvalue of type 'struct Flip'}}

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
  if (a) { } // expected-error {{conversion from 'class Yb' to 'bool' is ambiguous}}
  int i = a; // OK. calls XB::operator int();
  char ch = a;  // OK. calls Yb::operator char();
}

