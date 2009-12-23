// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks %s

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

int test_init_in_condition_aux();
int test_init_in_condition() {
  if (int x = test_init_in_condition_aux()) { // no-warning
    return 1;
  }
  return 0;
}
