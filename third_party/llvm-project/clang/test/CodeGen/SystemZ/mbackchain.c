// RUN: %clang_cc1 -mbackchain -triple s390x-linux -emit-llvm -o - %s | FileCheck %s

// CHECK: define{{.*}} void @foo() [[NUW:#[0-9]+]]
void foo(void) {
}

// CHECK: attributes [[NUW]] = { {{.*}} "backchain" {{.*}} }
