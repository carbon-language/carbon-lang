// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s


template <class T>
class A {
public:
   void f(T a) { }// expected-note {{must qualify identifier to find this declaration in dependent base class}}
   void g();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
};

template <class T>
class B : public A<T> {
public:
	void z(T a)
    {
       f(a); // expected-warning {{use of identifier 'f' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
       g(); // expected-warning {{use of identifier 'g' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    }
};

template class B<int>; // expected-note {{requested here}}
template class B<char>;

void test()
{
    B<int> b;
    b.z(3);
}

struct A2 {
  template<class T> void f(T) {
    XX; //expected-error {{use of undeclared identifier 'XX'}}
    A2::XX; //expected-error {{no member named 'XX' in 'A2'}}
  }
};
template void A2::f(int);

template<class T0>
struct A3 {
  template<class T1> void f(T1) {
    XX; //expected-error {{use of undeclared identifier 'XX'}}
  }
};
template void A3<int>::f(int);

template<class T0>
struct A4 {
  void f(char) {
    XX; //expected-error {{use of undeclared identifier 'XX'}}
  }
};
template class A4<int>;


namespace lookup_dependent_bases_id_expr {

template<class T> class A {
public:
  int var;
};


template<class T>
class B : public A<T> {
public:
  void f() {
    var = 3;
  }
};

template class B<int>;

}



namespace lookup_dependent_base_class_static_function {

template <class T>
class A {
public:
   static void static_func();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
   void func();// expected-note {{must qualify identifier to find this declaration in dependent base class}}
};


template <class T>
class B : public A<T> {
public:
  static void z2(){
    static_func();  // expected-warning {{use of identifier 'static_func' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    func(); // expected-warning {{use of identifier 'func' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-error {{call to non-static member function without an object argument}}
  }
};
template class B<int>; // expected-note {{requested here}}

} 



namespace lookup_dependent_base_class_default_argument {

template<class T>
class A {
public:
  static int f1(); // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
  int f2(); // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
};

template<class T>
class B : public A<T> {
public:
  void g1(int p = f1());// expected-warning {{use of identifier 'f1' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  void g2(int p = f2());// expected-warning {{use of identifier 'f2' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}} expected-error {{call to non-static member function without an object argument}}
};

void foo()
{
	B<int> b;
	b.g1(); // expected-note {{required here}}
	b.g2(); // expected-note {{required here}}
}

}


namespace lookup_dependent_base_class_friend {

template <class T>
class B {
public:
  static void g();  // expected-note {{must qualify identifier to find this declaration in dependent base class}} 
};

template <class T>
class A : public B<T> {
public:
  friend void foo(A<T> p){
    g(); // expected-warning {{use of identifier 'g' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  }
};

int main2()
{
  A<int> a;
  foo(a); // expected-note {{requested here}}
}

}


namespace lookup_dependent_base_no_typo_correction {

class C {
public:
  int m_hWnd;
};

template <class T>
class A : public T {
public:
  void f(int hWnd) {
    m_hWnd = 1;
  }
};

template class A<C>;

}

namespace PR12701 {

class A {};
class B {};

template <class T>
class Base {
 public:
  bool base_fun(void* p) { return false; }  // expected-note {{must qualify identifier to find this declaration in dependent base class}}
  operator T*() const { return 0; }
};

template <class T>
class Container : public Base<T> {
 public:
  template <typename S>
  bool operator=(const Container<S>& rhs) {
    return base_fun(rhs);  // expected-warning {{use of identifier 'base_fun' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
  }
};

void f() {
  Container<A> text_provider;
  Container<B> text_provider2;
  text_provider2 = text_provider;  // expected-note {{in instantiation of function template specialization}}
}

}  // namespace PR12701

namespace PR16014 {

struct A {
  int a;
  static int sa;
};
template <typename T> struct B : T {
  int     foo() { return a; }
  int    *bar() { return &a; }
  int     baz() { return T::a; }
  int T::*qux() { return &T::a; }
  static int T::*stuff() { return &T::a; }
  static int stuff1() { return T::sa; }
  static int *stuff2() { return &T::sa; }
};

template <typename T> struct C : T {
  int     foo() { return b; }      // expected-error {{no member named 'b' in 'PR16014::C<PR16014::A>'}}
  int    *bar() { return &b; }     // expected-error {{no member named 'b' in 'PR16014::C<PR16014::A>'}}
  int     baz() { return T::b; }   // expected-error {{no member named 'b' in 'PR16014::A'}}
  int T::*qux() { return &T::b; }  // expected-error {{no member named 'b' in 'PR16014::A'}}
  int T::*fuz() { return &U::a; }  // expected-error {{use of undeclared identifier 'U'}}
};

template struct B<A>;
template struct C<A>;  // expected-note-re 1+ {{in instantiation of member function 'PR16014::C<PR16014::A>::{{.*}}' requested here}}

template <typename T> struct D : T {
  struct Inner {
    int foo() {
      // FIXME: MSVC can find this in D's base T!  Even worse, if ::sa exists,
      // clang will use it instead.
      return sa; // expected-error {{use of undeclared identifier 'sa'}}
    }
  };
};
template struct D<A>;

}

namespace PR19233 {
template <class T>
struct A : T {
  void foo() {
    ::undef(); // expected-error {{no member named 'undef' in the global namespace}}
  }
  void bar() {
    ::UndefClass::undef(); // expected-error {{no member named 'UndefClass' in the global namespace}}
  }
  void baz() {
    B::qux(); // expected-error {{use of undeclared identifier 'B'}}
  }
};

struct B { void qux(); };
struct C : B { };
template struct A<C>; // No error!  B is a base of A<C>, and qux is available.

struct D { };
template struct A<D>; // expected-note {{in instantiation of member function 'PR19233::A<PR19233::D>::baz' requested here}}

}
