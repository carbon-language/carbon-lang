// RUN: %clang_cc1 -triple i386-unknown-unknown -fexceptions -emit-llvm %s -o - | FileCheck %s
int c(void) __attribute__((const));
int p(void) __attribute__((pure));
int t(void);

// CHECK: define i32 @_Z1fv() {{.*}} {
int f(void) {
  // CHECK: call i32 @_Z1cv() nounwind readnone
  // CHECK: call i32 @_Z1pv() nounwind readonly
  return c() + p() + t();
}

// CHECK: declare i32 @_Z1cv() #1
// CHECK: declare i32 @_Z1pv() #2
// CHECK: declare i32 @_Z1tv() #0

// CHECK: attributes #0 = { "target-features"={{.*}} }
// CHECK: attributes #1 = { nounwind readnone "target-features"={{.*}} }
// CHECK: attributes #2 = { nounwind readonly "target-features"={{.*}} }
