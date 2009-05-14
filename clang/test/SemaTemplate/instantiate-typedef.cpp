// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct add_pointer {
  typedef T* type; // expected-error{{'type' declared as a pointer to a reference}}
};

add_pointer<int>::type test1(int * ptr) { return ptr; }

add_pointer<float>::type test2(int * ptr) { 
  return ptr; // expected-error{{incompatible type returning 'int *', expected 'add_pointer<float>::type' (aka 'float *')}}
}

add_pointer<int&>::type // expected-note{{in instantiation of template class 'struct add_pointer<int &>' requested here}} \
// expected-error {{unknown type name 'type'}}
test3(); 
