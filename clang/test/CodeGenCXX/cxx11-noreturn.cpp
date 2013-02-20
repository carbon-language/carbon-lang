// RUN: %clang_cc1 -emit-llvm -std=c++11 %s -o - | FileCheck %s

int g();

// CHECK: _Z1fv(){{.*}} #0
[[noreturn]] int f() {
  while (g()) {}
}

// CHECK: attributes #0 = { noreturn nounwind "target-features"={{.*}} }
// CHECK: attributes #1 = { "target-features"={{.*}} }
// CHECK: attributes #2 = { noreturn nounwind }
