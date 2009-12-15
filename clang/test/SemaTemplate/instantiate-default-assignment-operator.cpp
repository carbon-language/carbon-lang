// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename> struct PassRefPtr { };
template<typename T> struct RefPtr {
  RefPtr& operator=(const RefPtr&) { int a[sizeof(T) ? -1 : -1];} // expected-error 2 {{array size is negative}}
  RefPtr& operator=(const PassRefPtr<T>&);
};

struct A { RefPtr<int> a; };
struct B : RefPtr<float> { };

void f() {
  A a1, a2;
  a1 = a2; // expected-note {{instantiation of member function 'RefPtr<int>::operator=' requested here}}

  B b1, b2;
  b1 = b2; // expected-note {{in instantiation of member function 'RefPtr<float>::operator=' requested here}}
}
