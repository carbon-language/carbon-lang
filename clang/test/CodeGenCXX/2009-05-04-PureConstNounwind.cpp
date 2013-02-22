// RUN: %clang_cc1 -triple i386-unknown-unknown -fexceptions -emit-llvm %s -o - | FileCheck %s
int c(void) __attribute__((const));
int p(void) __attribute__((pure));
int t(void);

// CHECK: define i32 @_Z1fv() {{.*}} {
int f(void) {
  // CHECK: call i32 @_Z1cv() [[NUW_RN:#[0-9]+]]
  // CHECK: call i32 @_Z1pv() [[NUW_RO:#[0-9]+]]
  return c() + p() + t();
}

// CHECK: declare i32 @_Z1cv() [[NUW_RN]]
// CHECK: declare i32 @_Z1pv() [[NUW_RO]]
// CHECK: declare i32 @_Z1tv() #0

// CHECK: attributes #0 = { "target-features"={{.*}} }
// CHECK: attributes [[NUW_RN]] = { nounwind readnone "target-features"={{.*}} }
// CHECK: attributes [[NUW_RO]] = { nounwind readonly "target-features"={{.*}} }
