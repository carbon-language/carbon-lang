// RUN: clang-cc %s

typedef char one_byte;
struct two_bytes { char data[2]; };

template<typename T> one_byte __is_class_check(int T::*);
template<typename T> two_bytes __is_class_check(...);

template<typename T> struct is_class {
  static const bool value = sizeof(__is_class_check<T>(0)) == 1;
};

struct X { };

int array0[is_class<X>::value? 1 : -1];
int array1[is_class<int>::value? -1 : 1];
int array2[is_class<char[3]>::value? -1 : 1];
