// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-nonfragile-abi -fblocks -fobjc-arc -O2 -disable-llvm-optzns -o - %s | FileCheck %s

id getObject();
void callee();

// Lifetime extension for binding a reference to an rvalue
// CHECK: define void @_Z5test0v()
void test0() {
  // CHECK: call i8* @_Z9getObjectv
  // CHECK-NEXT:: call i8* @objc_retainAutoreleasedReturnValue
  const __strong id &ref1 = getObject();
  // CHECK: call void @_Z6calleev
  callee();
  // CHECK: call i8* @_Z9getObjectv
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
  // CHECK-NEXT: call i8* @objc_autorelease
  const __autoreleasing id &ref2 = getObject();
  // CHECK: call void @_Z6calleev
  callee();
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret
}

// No lifetime extension when we're binding a reference to an lvalue.
// CHECK: define void @_Z5test1RU8__strongP11objc_objectRU6__weakS0_
void test1(__strong id &x, __weak id &y) {
  // CHECK-NOT: release
  const __strong id &ref1 = x;
  const __autoreleasing id &ref2 = x;
  const __weak id &ref3 = y;
  // CHECK: ret void
}

typedef __strong id strong_id;

//CHECK: define void @_Z5test3v
void test3() {
  // CHECK: call i8* @objc_initWeak
  // CHECK-NEXT: store i8**
  const __weak id &ref = strong_id();
  // CHECK-NEXT: call void @_Z6calleev()
  callee();
  // CHECK-NEXT: call void @objc_destroyWeak
  // CHECK-NEXT: ret void
}

// CHECK: define internal void @__cxx_global_var_init(
// CHECK: call i8* @_Z9getObjectv
// CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
const __strong id &global_ref = getObject();

// Note: we intentionally don't release the object.

