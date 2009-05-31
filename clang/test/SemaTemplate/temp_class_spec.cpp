// RUN: clang-cc -fsyntax-only -verify %s
template<typename T>
struct is_pointer {
  static const bool value = false;
};

template<typename T>
struct is_pointer<T*> {
  static const bool value = true;
};

template<typename T>
struct is_pointer<const T*> {
  static const bool value = true;
};

int array0[is_pointer<int>::value? -1 : 1];
int array1[is_pointer<int*>::value? 1 : -1];
int array2[is_pointer<const int*>::value? 1 : -1]; // expected-error{{partial ordering}} \
// expected-error{{negative}}
