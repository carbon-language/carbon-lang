// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 
// rdar://11124775

typedef signed char BOOL;

BOOL yes() {
  return __objc_yes;
}

BOOL no() {
  return __objc_no;
}

BOOL which (int flag) {
  return flag ? yes() : no();
}

int main() {
  which (__objc_yes);
  which (__objc_no);
  return __objc_yes;
}

// CHECK: return ((signed char)1);
// CHECK: return ((signed char)0);
// CHECK: which (((signed char)1));
// CHECK: which (((signed char)0));
// CHECK: return ((signed char)1);
