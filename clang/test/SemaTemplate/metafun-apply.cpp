// RUN: clang-cc -fsyntax-only -verify %s

struct add_pointer {
  template<typename T>
  struct apply {
    typedef T* type;
  };
};

struct add_reference {
  template<typename T>
  struct apply {
    typedef T& type; // expected-error{{cannot form a reference to 'void'}}
  };
};

struct bogus {
  struct apply {
    typedef int type;
  };
};

template<typename MetaFun, typename T>
struct apply1 {
  typedef typename MetaFun::template apply<T>::type type; // expected-note{{in instantiation of template class 'struct add_reference::apply<void>' requested here}} \
  // expected-error{{'apply' following the 'template' keyword does not refer to a template}} \
  // FIXME: expected-error{{type 'MetaFun::template apply<int>' cannot be used prior to '::' because it has no members}}
};

int i;
apply1<add_pointer, int>::type ip = &i;
apply1<add_reference, int>::type ir = i;
apply1<add_reference, float>::type fr = i; // expected-error{{non-const lvalue reference to type 'float' cannot be initialized with a value of type 'int'}}

void test() {
  apply1<add_reference, void>::type t; // expected-note{{in instantiation of template class 'struct apply1<struct add_reference, void>' requested here}} \
  // FIXME: expected-error{{unexpected type name 'type': expected expression}}

  apply1<bogus, int>::type t2; // expected-note{{in instantiation of template class 'struct apply1<struct bogus, int>' requested here}} \
  // FIXME: expected-error{{unexpected type name 'type': expected expression}}
}


