// RUN: %clang_cc1 -triple i386-unknown-unknown -fexceptions -emit-llvm %s -o - | FileCheck %s
int c(void) __attribute__((const));
int p(void) __attribute__((pure));
int t(void);

// CHECK: define i32 @_Z1fv() {{.*}} {
int f(void) {
  // CHECK: call i32 @_Z1cv() [[NUW_RN1:#[0-9]+]]
  // CHECK: call i32 @_Z1pv() [[NUW_RO1:#[0-9]+]]
  return c() + p() + t();
}

// CHECK: declare i32 @_Z1cv() [[NUW_RN2:#[0-9]+]]
// CHECK: declare i32 @_Z1pv() [[NUW_RO2:#[0-9]+]]
// CHECK: declare i32 @_Z1tv() [[NONE:#[0-9]+]]

// CHECK: attributes [[NONE]] = { {{.*}} }
// CHECK: attributes [[NUW_RN2]] = { nounwind readnone{{.*}} }
// CHECK: attributes [[NUW_RO2]] = { nounwind readonly{{.*}} }
// CHECK: attributes [[NUW_RN1]] = { nounwind readnone }
// CHECK: attributes [[NUW_RO1]] = { nounwind readonly }
