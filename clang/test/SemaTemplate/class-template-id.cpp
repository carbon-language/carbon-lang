// RUN: clang -fsyntax-only -verify %s
template<typename T> struct A { };

typedef A<int> A_int;

float *foo(A<int> *ptr, A<int> const *ptr2) {
  if (ptr)
    return ptr; // expected-error{{incompatible type returning 'A<int> *', expected 'float *'}}
  else if (ptr2)
    return ptr2; // expected-error{{incompatible type returning 'A<int> const *', expected 'float *'}}
  else {
    // FIXME: This is completely bogus, but we're using it temporarily
    // to test the syntactic sugar for class template specializations.
    int *ip = ptr;
    return 0;
  }
}
