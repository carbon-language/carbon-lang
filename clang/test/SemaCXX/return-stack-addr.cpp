// RUN: %clang_cc1 -fsyntax-only -verify %s

int* ret_local() {
  int x = 1;
  return &x; // expected-warning {{address of stack memory}}
}

int* ret_local_array() {
  int x[10];
  return x; // expected-warning {{address of stack memory}}
}

int* ret_local_array_element(int i) {
  int x[10];
  return &x[i]; // expected-warning {{address of stack memory}}
}

int *ret_local_array_element_reversed(int i) {
  int x[10];
  return &i[x]; // expected-warning {{address of stack memory}}
}

int* ret_local_array_element_const_index() {
  int x[10];
  return &x[2];  // expected-warning {{address of stack memory}}
}

int& ret_local_ref() {
  int x = 1;
  return x;  // expected-warning {{reference to stack memory}}
}

int* ret_local_addrOf() {
  int x = 1;
  return &*&x; // expected-warning {{address of stack memory}}
}

int* ret_local_addrOf_paren() {
  int x = 1;
  return (&(*(&x))); // expected-warning {{address of stack memory}}
}

int* ret_local_addrOf_ptr_arith() {
  int x = 1;
  return &*(&x+1); // expected-warning {{address of stack memory}}
}

int* ret_local_addrOf_ptr_arith2() {
  int x = 1;
  return &*(&x+1); // expected-warning {{address of stack memory}}
}

int* ret_local_field() {
  struct { int x; } a;
  return &a.x; // expected-warning {{address of stack memory}}
}

int& ret_local_field_ref() {
  struct { int x; } a;
  return a.x; // expected-warning {{reference to stack memory}}
}

int* ret_conditional(bool cond) {
  int x = 1;
  int y = 2;
  return cond ? &x : &y; // expected-warning {{address of stack memory}}
}

int* ret_conditional_rhs(int *x, bool cond) {
  int y = 1;
  return cond ? x : &y;  // expected-warning {{address of stack memory}}
}

void* ret_c_cast() {
  int x = 1;
  return (void*) &x;  // expected-warning {{address of stack memory}}
}

int* ret_static_var() {
  static int x = 1;
  return &x;  // no warning.
}

int z = 1;

int* ret_global() {
  return &z;  // no warning.
}

int* ret_parameter(int x) {
  return &x;  // expected-warning {{address of stack memory}}
}


void* ret_cpp_static_cast(short x) {
  return static_cast<void*>(&x); // expected-warning {{address of stack memory}}
}

int* ret_cpp_reinterpret_cast(double x) {
  return reinterpret_cast<int*>(&x); // expected-warning {{address of stack me}}
}

int* ret_cpp_reinterpret_cast_no_warning(long x) {
  return reinterpret_cast<int*>(x); // no-warning
}

int* ret_cpp_const_cast(const int x) {
  return const_cast<int*>(&x);  // expected-warning {{address of stack memory}}
}

// TODO: test case for dynamic_cast.  clang does not yet have
// support for C++ classes to write such a test case.
