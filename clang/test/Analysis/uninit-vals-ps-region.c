// RUN: %clang_cc1 -analyze -checker-cfref -analyzer-store=region -verify %s

struct s {
  int data;
};

struct s global;

void g(int);

void f4() {
  int a;
  if (global.data == 0)
    a = 3;
  if (global.data == 0) // When the true branch is feasible 'a = 3'.
    g(a); // no-warning
}


// Test uninitialized value due to part of the structure being uninitialized.
struct TestUninit { int x; int y; };
struct TestUninit test_uninit_aux();
void test_uninit_pos() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2 = test_uninit_aux();
  int z;
  v1.y = z; // expected-warning{{Assigned value is garbage or undefined}}
  test_unit_aux2(v2.x + v1.y);
}
void test_uninit_pos_2() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2;
  test_unit_aux2(v2.x + v1.y);  // expected-warning{{The left operand of '+' is a garbage value}}
}
void test_uninit_pos_3() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2;
  test_unit_aux2(v1.y + v2.x);  // expected-warning{{The right operand of '+' is a garbage value}}
}

void test_uninit_neg() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2 = test_uninit_aux();
  test_unit_aux2(v2.x + v1.y); // no-warning
}

