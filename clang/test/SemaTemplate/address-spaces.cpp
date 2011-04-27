// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

typedef int __attribute__((address_space(1))) int_1;;
typedef int __attribute__((address_space(2))) int_2;;
typedef int __attribute__((address_space(1))) *int_1_ptr;
typedef int_2 *int_2_ptr;

// Check that we maintain address spaces through template argument
// deduction from a type.
template<typename T>
struct remove_pointer {
  typedef T type;
};

template<typename T>
struct remove_pointer<T *> {
  typedef T type;
};

int check_remove0[is_same<remove_pointer<int_1_ptr>::type, int_1>::value? 1 : -1];
int check_remove1[is_same<remove_pointer<int_2_ptr>::type, int_2>::value? 1 : -1];
int check_remove2[is_same<remove_pointer<int_2_ptr>::type, int>::value? -1 : 1];
int check_remove3[is_same<remove_pointer<int_2_ptr>::type, int_1>::value? -1 : 1];
                
// Check that we maintain address spaces through template argument
// deduction for a call.
template<typename T>
void accept_any_pointer(T*) {
  T *x = 1; // expected-error{{cannot initialize a variable of type '__attribute__((address_space(1))) int *' with an rvalue of type 'int'}} \
  // expected-error{{cannot initialize a variable of type '__attribute__((address_space(3))) int *' with an rvalue of type 'int'}}
}

void test_accept_any_pointer(int_1_ptr ip1, int_2_ptr ip2) {
  static __attribute__((address_space(3))) int array[17];
  accept_any_pointer(ip1); // expected-note{{in instantiation of}}
  accept_any_pointer(array); // expected-note{{in instantiation of}}
}

