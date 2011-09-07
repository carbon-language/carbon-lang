// RUN: %clang_cc1 -fms-extensions -fsyntax-only -verify %s


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


