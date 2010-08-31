// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks %s

// Test basic handling of references.
char &test1_aux();
char *test1() {
  return &test1_aux();
}

// Test test1_aux() evaluates to char &.
char test1_as_rvalue() {
  return test1_aux();
}

// Test passing a value as a reference.  The 'const' in test2_aux() adds
// an ImplicitCastExpr, which is evaluated as an lvalue.
int test2_aux(const int &n);
int test2(int n) {
  return test2_aux(n);
}

int test2_b_aux(const short &n);
int test2_b(int n) {
  return test2_b_aux(n);
}

// Test getting the lvalue of a derived and converting it to a base.  This
// previously crashed.
class Test3_Base {};
class Test3_Derived : public Test3_Base {};

int test3_aux(Test3_Base &x);
int test3(Test3_Derived x) {
  return test3_aux(x);
}

//===---------------------------------------------------------------------===//
// Test CFG support for C++ condition variables.
//===---------------------------------------------------------------------===//

int test_init_in_condition_aux();
int test_init_in_condition() {
  if (int x = test_init_in_condition_aux()) { // no-warning
    return 1;
  }
  return 0;
}

int test_init_in_condition_switch() {
  switch (int x = test_init_in_condition_aux()) { // no-warning
    case 1:
      return 0;
    case 2:
      if (x == 2)
        return 0;
      else {
        // Unreachable.
        int *p = 0;
        *p = 0xDEADBEEF; // no-warning
      }
    default:
      break;
  }
  return 0;
}

int test_init_in_condition_while() {
  int z = 0;
  while (int x = ++z) { // no-warning
    if (x == 2)
      break;
  }
  
  if (z == 2)
    return 0;
  
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 0;
}


int test_init_in_condition_for() {
  int z = 0;
  for (int x = 0; int y = ++z; ++x) {
    if (x == y) // no-warning
      break;
  }
  if (z == 1)
    return 0;
    
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 0;
}

//===---------------------------------------------------------------------===//
// Test handling of 'this' pointer.
//===---------------------------------------------------------------------===//

class TestHandleThis {
  int x;

  TestHandleThis();  
  int foo();
  int null_deref_negative();
  int null_deref_positive();  
};

int TestHandleThis::foo() {
  // Assume that 'x' is initialized.
  return x + 1; // no-warning
}

int TestHandleThis::null_deref_negative() {
  x = 10;
  if (x == 10) {
    return 1;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 0;  
}

int TestHandleThis::null_deref_positive() {
  x = 10;
  if (x == 9) {
    return 1;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{null pointer}}
  return 0;  
}

// PR 7675 - passing literals by-reference
void pr7675(const double &a);
void pr7675(const int &a);
void pr7675(const char &a);
void pr7675_i(const _Complex double &a);

void pr7675_test() {
  pr7675(10.0);
  pr7675(10);
  pr7675('c');
  pr7675_i(4.0i);
  // Add null deref to ensure we are analyzing the code up to this point.
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{null pointer}}
}

// <rdar://problem/8375510> - CFGBuilder should handle temporaries.
struct R8375510 {
  R8375510();
  ~R8375510();
  R8375510 operator++(int);
};

int r8375510(R8375510 x, R8375510 y) {
  for (; ; x++) { }
}

