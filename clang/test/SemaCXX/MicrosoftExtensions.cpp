// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wmicrosoft -Wc++11-extensions -Wno-long-long -verify -fms-extensions -fexceptions -fcxx-exceptions


// ::type_info is predeclared with forward class declartion
void f(const type_info &a);


// Microsoft doesn't validate exception specification.
namespace microsoft_exception_spec {

void foo(); // expected-note {{previous declaration}}
void foo() throw(); // expected-warning {{exception specification in declaration does not match previous declaration}}

void r6() throw(...); // expected-note {{previous declaration}}
void r6() throw(int); // expected-warning {{exception specification in declaration does not match previous declaration}}

struct Base {
  virtual void f2();
  virtual void f3() throw(...);
};

struct Derived : Base {
  virtual void f2() throw(...);
  virtual void f3();
};

class A {
  virtual ~A() throw();  // expected-note {{overridden virtual function is here}}
};

class B : public A {
  virtual ~B();  // expected-warning {{exception specification of overriding function is more lax than base version}}
};

}

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

// __unaligned handling
typedef char __unaligned *aligned_type;


template<typename T> void h1(T (__stdcall M::* const )()) { }

void m1() {
  h1<int>(&M::addP);
  h1(&M::subtractP);
} 





void f(long long);
void f(int);
 
int main()
{
  // This is an ambiguous call in standard C++.
  // This calls f(long long) in Microsoft mode because LL is always signed.
  f(0xffffffffffffffffLL);
  f(0xffffffffffffffffi64);
}

// Enumeration types with a fixed underlying type.
const int seventeen = 17;
typedef int Int;

struct X0 {
  enum E1 : Int { SomeOtherValue } field; // expected-warning{{enumeration types with a fixed underlying type are a C++11 extension}}
  enum E1 : seventeen;
};

enum : long long {  // expected-warning{{enumeration types with a fixed underlying type are a C++11 extension}}
  SomeValue = 0x100000000
};


class AAA {
__declspec(dllimport) void f(void) { }
void f2(void);
};

__declspec(dllimport) void AAA::f2(void) { // expected-error {{dllimport attribute can be applied only to symbol}}

}



template <class T>
class BB {
public:
   void f(int g = 10 ); // expected-note {{previous definition is here}}
};

template <class T>
void BB<T>::f(int g = 0) { } // expected-warning {{redefinition of default argument}}



extern void static_func();
void static_func(); // expected-note {{previous declaration is here}}


static void static_func() // expected-warning {{static declaration of 'static_func' follows non-static declaration}}
{

}

long function_prototype(int a);
long (*function_ptr)(int a);

void function_to_voidptr_conv() {
   void *a1 = function_prototype;
   void *a2 = &function_prototype;
   void *a3 = function_ptr;
}


void pointer_to_integral_type_conv(char* ptr) {
   char ch = (char)ptr;
   short sh = (short)ptr;
   ch = (char)ptr;
   sh = (short)ptr;
} 


namespace friend_as_a_forward_decl {

class A {
  class Nested {
    friend class B;
    B* b;
  };
  B* b;
};
B* global_b;


void f()
{
  class Local {
    friend class Z;
    Z* b;
  };
  Z* b;
}

}

struct PR11150 {
  class X {
    virtual void f() = 0;
  };

  int array[__is_abstract(X)? 1 : -1];
};

void f() { int __except = 0; }

void ::f(); // expected-warning{{extra qualification on member 'f'}}

class C {
  C::C(); // expected-warning{{extra qualification on member 'C'}}
};

struct StructWithProperty {
  __declspec(property(get=GetV)) int V1;
  __declspec(property(put=SetV)) int V2;
  __declspec(property(get=GetV, put=SetV_NotExist)) int V3;
  __declspec(property(get=GetV_NotExist, put=SetV)) int V4;
  __declspec(property(get=GetV, put=SetV)) int V5;

  int GetV() { return 123; }
  void SetV(int i) {}
};
void TestProperty() {
  StructWithProperty sp;
  int i = sp.V2; // expected-error{{no getter defined for property 'V2'}}
  sp.V1 = 12; // expected-error{{no setter defined for property 'V1'}}
  int j = sp.V4; // expected-error{{no member named 'GetV_NotExist' in 'StructWithProperty'}} expected-error{{cannot find suitable getter for property 'V4'}}
  sp.V3 = 14; // expected-error{{no member named 'SetV_NotExist' in 'StructWithProperty'}} expected-error{{cannot find suitable setter for property 'V3'}}
  int k = sp.V5;
  sp.V5 = k++;
}

/* 4 tests for PseudoObject, begin */
struct SP1
{
  bool operator()() { return true; }
};
struct SP2
{
  __declspec(property(get=GetV)) SP1 V;
  SP1 GetV() { return SP1(); }
};
void TestSP2() {
  SP2 sp2;
  bool b = sp2.V();
}

struct SP3 {
  template <class T>
  void f(T t) {}
};
template <class T>
struct SP4
{
  __declspec(property(get=GetV)) int V;
  int GetV() { return 123; }
  void f() { SP3 s2; s2.f(V); }
};
void TestSP4() {
  SP4<int> s;
  s.f();
}

template <class T>
struct SP5
{
  __declspec(property(get=GetV)) T V;
  int GetV() { return 123; }
  void f() { int *p = new int[V]; }
};

template <class T>
struct SP6
{
public:
  __declspec(property(get=GetV)) T V;
  T GetV() { return 123; }
  void f() { int t = V; }
};
void TestSP6() {
  SP6<int> c;
  c.f();
}
/* 4 tests for PseudoObject, end */

// Property access: explicit, implicit, with Qualifier
struct SP7 {
  __declspec(property(get=GetV, put=SetV)) int V;
  int GetV() { return 123; }
  void SetV(int v) {}

  void ImplicitAccess() { int i = V; V = i; }
  void ExplicitAccess() { int i = this->V; this->V = i; }
};
struct SP8: public SP7 {
  void AccessWithQualifier() { int i = SP7::V; SP7::V = i; }
};

// Property usage
template <class T>
struct SP9 {
  __declspec(property(get=GetV, put=SetV)) T V;
  T GetV() { return 0; }
  void SetV(T v) {}
  void f() { V = this->V; V < this->V; }
  void g() { V++; }
  void h() { V*=2; }
};
struct SP10 {
  SP10(int v) {}
  bool operator<(const SP10& v) { return true; }
  SP10 operator*(int v) { return *this; }
  SP10 operator+(int v) { return *this; }
  SP10& operator=(const SP10& v) { return *this; }
};
void TestSP9() {
  SP9<int> c;
  int i = c.V; // Decl initializer
  i = c.V; // Binary op operand
  c.SetV(c.V); // CallExpr arg
  int *p = new int[c.V + 1]; // Array size
  p[c.V] = 1; // Array index

  c.V = 123; // Setter

  c.V++; // Unary op operand
  c.V *= 2; // Unary op operand

  SP9<int*> c2;
  c2.V[0] = 123; // Array

  SP9<SP10> c3;
  c3.f(); // Overloaded binary op operand
  c3.g(); // Overloaded incdec op operand
  c3.h(); // Overloaded unary op operand
}
