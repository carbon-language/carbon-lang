// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions -fexceptions


// ::type_info is predeclared with forward class declartion
void f(const type_info &a);

// The following three are all equivalent when ms-extensions are on
void foo() throw(int);
void foo() throw(int, long);
void foo() throw(...); 
void foo(); // expected-note {{previous declaration}}

// Only nothrow specification is treated specially.
void foo() throw(); // expected-error {{exception specification in declaration does not match previous declaration}}

// throw(...)
void r3();
void r3() throw(...);

void r6() throw(...);
void r6() throw(int); // okay

struct Base {
  virtual void f2();
  virtual void f3() throw(...);
};

struct Derived : Base {
  virtual void f2() throw(...);
  virtual void f3();
};


// MSVC allows type definition in anonymous union and struct
struct A
{
  union 
  {
    int a;
    struct B  // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
    { 
      int c;
    } d;

    union C   // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
    {
      int e;
      int ee;
    } f;

    typedef int D;  // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
    struct F;  // expected-warning {{types declared in an anonymous union are a Microsoft extension}}
  };

  struct
  {
    int a2;

    struct B2  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
    {
      int c2;
    } d2;
    
	union C2  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
    {
      int e2;
      int ee2;
    } f2;

    typedef int D2;  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
    struct F2;  // expected-warning {{types declared in an anonymous struct are a Microsoft extension}}
  };
};

// __stdcall handling
struct M {
    int __stdcall addP();
    float __stdcall subtractP(); 
};

template<typename T> void h1(T (__stdcall M::* const )()) { }

void m1() {
  h1<int>(&M::addP);
  h1(&M::subtractP);
} 

//MSVC allows forward enum declaration
enum ENUM; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}
ENUM *var = 0;     
ENUM var2 = (ENUM)3;
enum ENUM1* var3 = 0;// expected-warning {{forward references to 'enum' types are a Microsoft extension}}


enum ENUM2 {
	ENUM2_a = (enum ENUM2) 4,
	ENUM2_b = 0x9FFFFFFF, // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
	ENUM2_c = 0x100000000 // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
};
