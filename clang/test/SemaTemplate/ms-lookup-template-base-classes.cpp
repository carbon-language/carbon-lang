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
