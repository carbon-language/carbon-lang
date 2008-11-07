// RUN: clang -fsyntax-only -verify %s 
class X { 
public:
  operator bool();
  operator int() const;
};

operator int(); // expected-error{{conversion function must be a non-static member function}}

typedef int func_type(int);
typedef int array_type[10];

class Y {
public:
  void operator bool(int, ...) const; // expected-error{{conversion function cannot have a return type}} \
  // expected-error{{conversion function cannot have any parameters}} \
  // expected-error{{conversion function cannot be variadic}}
  operator func_type(); // expected-error{{conversion function cannot convert to a function type}}
  operator array_type(); // expected-error{{conversion function cannot convert to an array type}}
};


typedef int INT;
typedef INT* INT_PTR;

class Z { 
  operator int(); // expected-error{{previous declaration is here}}
  operator int**(); // expected-error{{previous declaration is here}}
  
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
