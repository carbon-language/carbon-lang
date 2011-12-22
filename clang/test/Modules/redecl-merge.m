// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -I %S/Inputs %s -verify
// RUN: %clang_cc1 -x objective-c++ -fmodule-cache-path %t -I %S/Inputs %s -verify
@class C2;
@class C3;
@class C3;
__import_module__ redecl_merge_left;

@class C3;
@class C3;
__import_module__ redecl_merge_right;

@implementation A
- (Super*)init { return self; }
@end

void f(A *a) {
  [a init];
}

@class A;

B *f1() {
  return [B create_a_B];
}

@class B;

// Test redeclarations of entities in explicit submodules, to make
// sure we're maintaining the declaration chains even when normal name
// lookup can't see what we're looking for.
void testExplicit() {
  Explicit *e;
  int *(*fp)(void) = &explicit_func;
  int *ip = explicit_func();

  // FIXME: Should complain about definition not having been imported.
  struct explicit_struct es = { 0 };
}

// Test resolution of declarations from multiple modules with no
// common original declaration.
void test_C(C *c) {
  c = get_a_C();
  accept_a_C(c);
}

void test_C2(C2 *c2) {
  c2 = get_a_C2();
  accept_a_C2(c2);
}

void test_C3(C3 *c3) {
  c3 = get_a_C3();
  accept_a_C3(c3);
}

C4 *global_C4;
__import_module__ redecl_merge_left_left;

void test_C4(C4 *c4) {
  global_C4 = c4 = get_a_C4();
  accept_a_C4(c4);
}

__import_module__ redecl_merge_bottom;

@implementation B
+ (B*)create_a_B { return 0; }
@end

void g(A *a) {
  [a init];
}

#ifdef __cplusplus
void testVector() {
  Vector<int> vec_int;
  vec_int.push_back(0);
}
#endif

