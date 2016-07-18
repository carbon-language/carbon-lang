// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -main-file-name abspath.cpp %S/Inputs/../abspath.cpp -o - | FileCheck %s

// CHECK: @__llvm_coverage_mapping = {{.*}}"\01
// CHECK-NOT: Inputs
// CHECK: "

void f1() {}
