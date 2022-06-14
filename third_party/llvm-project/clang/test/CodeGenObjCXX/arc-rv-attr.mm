// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-ios9 -fobjc-runtime=ios-9.0 -fobjc-arc -std=c++11 -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK

id foo(void);

// CHECK-LABEL: define{{.*}} void @_Z14test_list_initv(
// CHECK: %[[CALL1:.*]] = call noundef i8* @_Z3foov() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK: call i8* @llvm.objc.retain(i8* %[[CALL1]])

void test_list_init() {
  auto t = id{foo()};
}
