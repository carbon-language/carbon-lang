// RUN: %clang_cc1 -verify %s

int foo() {
  int x[2]; // expected-note 4 {{array 'x' declared here}}
  int y[2]; // expected-note 2 {{array 'y' declared here}}
  int *p = &y[2]; // no-warning
  (void) sizeof(x[2]); // no-warning
  y[2] = 2; // expected-warning {{array index of '2' indexes past the end of an array (that contains 2 elements)}}
  return x[2] +  // expected-warning {{array index of '2' indexes past the end of an array (that contains 2 elements)}}
         y[-1] + // expected-warning {{array index of '-1' indexes before the beginning of the array}}
         x[sizeof(x)] +  // expected-warning {{array index of '8' indexes past the end of an array (that contains 2 elements)}}
         x[sizeof(x) / sizeof(x[0])] +  // expected-warning {{array index of '2' indexes past the end of an array (that contains 2 elements)}}
         x[sizeof(x) / sizeof(x[0]) - 1] + // no-warning
         x[sizeof(x[2])]; // expected-warning {{array index of '4' indexes past the end of an array (that contains 2 elements)}}
}

// This code example tests that -Warray-bounds works with arrays that
// are template parameters.
template <char *sz> class Qux {
  bool test() { return sz[0] == 'a'; }
};

void f1(int a[1]) {
  int val = a[3]; // no warning for function argumnet
}

void f2(const int (&a)[1]) { // expected-note {{declared here}}
  int val = a[3];  // expected-warning {{array index of '3' indexes past the end of an array (that contains 1 element)}}
}

void test() {
  struct {
    int a[0];
  } s2;
  s2.a[3] = 0; // no warning for 0-sized array

  union {
    short a[2]; // expected-note 2 {{declared here}}
    char c[4];
  } u;
  u.a[3] = 1; // expected-warning {{array index of '3' indexes past the end of an array (that contains 2 elements)}}
  u.c[3] = 1; // no warning
  *(&u.a[3]) = 1; // expected-warning {{array index of '3' indexes past the end of an array (that contains 2 elements)}}
  *(&u.c[3]) = 1; // no warning

  const int const_subscript = 3;
  int array[1]; // expected-note {{declared here}}
  array[const_subscript] = 0;  // expected-warning {{array index of '3' indexes past the end of an array (that contains 1 element)}}

  int *ptr;
  ptr[3] = 0; // no warning for pointer references
  int array2[] = { 0, 1, 2 }; // expected-note 2 {{declared here}}

  array2[3] = 0; // expected-warning {{array index of '3' indexes past the end of an array (that contains 3 elements)}}
  array2[2+2] = 0; // expected-warning {{array index of '4' indexes past the end of an array (that contains 3 elements)}}

  const char *str1 = "foo";
  char c1 = str1[5]; // no warning for pointers

  const char str2[] = "foo"; // expected-note {{declared here}}
  char c2 = str2[5]; // expected-warning {{array index of '5' indexes past the end of an array (that contains 4 elements)}}

  int (*array_ptr)[1];
  (*array_ptr)[3] = 1; // expected-warning {{array index of '3' indexes past the end of an array (that contains 1 element)}}
}

template <int I> struct S {
  char arr[I]; // expected-note 2 {{declared here}}
};
template <int I> void f() {
  S<3> s;
  s.arr[4] = 0; // expected-warning {{array index of '4' indexes past the end of an array (that contains 3 elements)}}
  s.arr[I] = 0; // expected-warning {{array index of '5' indexes past the end of an array (that contains 3 elements)}}
}

void test_templates() {
  f<5>(); // expected-note {{in instantiation}}
}

#define SIZE 10
#define ARR_IN_MACRO(flag, arr, idx) flag ? arr[idx] : 1

int test_no_warn_macro_unreachable() {
  int arr[SIZE]; // expected-note {{array 'arr' declared here}}
  return ARR_IN_MACRO(0, arr, SIZE) + // no-warning
         ARR_IN_MACRO(1, arr, SIZE); // expected-warning{{array index of '10' indexes past the end of an array (that contains 10 elements)}}
}

// This exhibited an assertion failure for a 32-bit build of Clang.
int test_pr9240() {
  short array[100]; // expected-note {{array 'array' declared here}}
  return array[(unsigned long long) 100]; // expected-warning {{array index of '100' indexes past the end of an array (that contains 100 elements)}}
}

// PR 9284 - a template parameter can cause an array bounds access to be
// infeasible.
template <bool extendArray>
void pr9284() {
    int arr[3 + (extendArray ? 1 : 0)];

    if (extendArray)
        arr[3] = 42; // no-warning
}

template <bool extendArray>
void pr9284b() {
    int arr[3 + (extendArray ? 1 : 0)]; // expected-note {{array 'arr' declared here}}

    if (!extendArray)
        arr[3] = 42; // expected-warning{{array index of '3' indexes past the end of an array (that contains 3 elements)}}
}

void test_pr9284() {
    pr9284<true>();
    pr9284<false>();
    pr9284b<true>();
    pr9284b<false>(); // expected-note{{in instantiation of function template specialization 'pr9284b<false>' requested here}}
}

int test_pr9296() {
    int array[2];
    return array[true]; // no-warning
}

int test_sizeof_as_condition(int flag) {
  int arr[2] = { 0, 0 }; // expected-note {{array 'arr' declared here}}
  if (flag) 
    return sizeof(char) != sizeof(char) ? arr[2] : arr[1];
  return sizeof(char) == sizeof(char) ? arr[2] : arr[1]; // expected-warning {{array index of '2' indexes past the end of an array (that contains 2 elements)}}
}

void test_switch() {
  switch (4) {
    case 1: {
      int arr[2];
      arr[2] = 1; // no-warning
      break;
    }
    case 4: {
      int arr[2]; // expected-note {{array 'arr' declared here}}
      arr[2] = 1; // expected-warning {{array index of '2' indexes past the end of an array (that contains 2 elements)}}
      break;
    }
    default: {
      int arr[2];
      arr[2] = 1; // no-warning
      break;
    }
  }
}

// Test nested switch statements.
enum enumA { enumA_A, enumA_B, enumA_C, enumA_D, enumA_E };
enum enumB { enumB_X, enumB_Y, enumB_Z };
static enum enumB myVal = enumB_X;
void test_nested_switch() {
  switch (enumA_E) { // expected-warning {{no case matching constant}}
    switch (myVal) { // expected-warning {{enumeration values 'enumB_X' and 'enumB_Z' not handled in switch}}
      case enumB_Y: ;
    }
  }
}

// Test that if all the values of an enum covered, that the 'default' branch
// is unreachable.
enum Values { A, B, C, D };
void test_all_enums_covered(enum Values v) {
  int x[2];
  switch (v) {
  case A: return;
  case B: return;
  case C: return;
  case D: return;
  }
  x[2] = 0; // no-warning
}
