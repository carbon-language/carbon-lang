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

template<typename T>
struct is_lvalue_reference {
  static const bool value = false;
};

template<typename T>
struct is_lvalue_reference<T&> {
  static const bool value = true;
};

int lvalue_ref0[is_lvalue_reference<int>::value? -1 : 1];
int lvalue_ref1[is_lvalue_reference<const int&>::value? 1 : -1];

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

template<typename T>
struct remove_reference {
  typedef T type;
};

template<typename T>
struct remove_reference<T&> {
  typedef T type;
};

int remove_ref0[is_same<remove_reference<int>::type, int>::value? 1 : -1];
int remove_ref1[is_same<remove_reference<int&>::type, int>::value? 1 : -1];
                
template<typename T>
struct is_incomplete_array {
  static const bool value = false;
};

template<typename T>
struct is_incomplete_array<T[]> {
  static const bool value = true;
};

int incomplete_array0[is_incomplete_array<int>::value ? -1 : 1];
int incomplete_array1[is_incomplete_array<int[1]>::value ? -1 : 1];
int incomplete_array2[is_incomplete_array<bool[]>::value ? 1 : -1];
int incomplete_array3[is_incomplete_array<int[]>::value ? 1 : -1];

template<typename T>
struct is_array_with_4_elements {
  static const bool value = false;
};

template<typename T>
struct is_array_with_4_elements<T[4]> {
  static const bool value = true;
};

int array_with_4_elements0[is_array_with_4_elements<int[]>::value ? -1 : 1];
int array_with_4_elements1[is_array_with_4_elements<int[1]>::value ? -1 : 1];
int array_with_4_elements2[is_array_with_4_elements<int[4]>::value ? 1 : -1];
int array_with_4_elements3[is_array_with_4_elements<int[4][2]>::value ? 1 : -1];

template<typename T>
struct get_array_size;

template<typename T, unsigned N>
struct get_array_size<T[N]> {
  static const unsigned value = N;
};

int array_size0[get_array_size<int[12]>::value == 12? 1 : -1];

template<typename T>
struct is_unary_function {
  static const bool value = false;
};

template<typename T, typename U>
struct is_unary_function<T (*)(U)> {
  static const bool value = true;
};

int is_unary_function0[is_unary_function<int>::value ? -1 : 1];
int is_unary_function1[is_unary_function<int (*)()>::value ? -1 : 1];
int is_unary_function2[is_unary_function<int (*)(int, bool)>::value ? -1 : 1];
int is_unary_function3[is_unary_function<int (*)(bool)>::value ? 1 : -1];
int is_unary_function4[is_unary_function<int (*)(int)>::value ? 1 : -1];

template<typename T>
struct is_unary_function_with_same_return_type_as_argument_type {
  static const bool value = false;
};

template<typename T>
struct is_unary_function_with_same_return_type_as_argument_type<T (*)(T)> {
  static const bool value = true;
};

int is_unary_function5[is_unary_function_with_same_return_type_as_argument_type<int>::value ? -1 : 1];
int is_unary_function6[is_unary_function_with_same_return_type_as_argument_type<int (*)()>::value ? -1 : 1];
int is_unary_function7[is_unary_function_with_same_return_type_as_argument_type<int (*)(int, bool)>::value ? -1 : 1];
int is_unary_function8[is_unary_function_with_same_return_type_as_argument_type<int (*)(bool)>::value ? -1 : 1];
int is_unary_function9[is_unary_function_with_same_return_type_as_argument_type<int (*)(int)>::value ? 1 : -1];

template<typename T>
struct is_binary_function {
  static const bool value = false;
};

template<typename R, typename T1, typename T2>
struct is_binary_function<R(T1, T2)> {
  static const bool value = true;
};

int is_binary_function0[is_binary_function<int(float, double)>::value? 1 : -1];
