// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodule-cache-path %t -I %S/Inputs %s -verify
// RUN: %clang_cc1 -x objective-c++ -fmodule-cache-path %t -I %S/Inputs %s -verify
__import_module__ redecl_merge_left;
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
