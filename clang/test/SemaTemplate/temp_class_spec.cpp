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

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

typedef int INT;
typedef INT* int_ptr;

int is_same0[is_same<int, int>::value? 1 : -1];
int is_same1[is_same<int, INT>::value? 1 : -1];
int is_same2[is_same<const int, int>::value? -1 : 1];
int is_same3[is_same<int_ptr, int>::value? -1 : 1];
