// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -o - -test-coverage -femit-coverage-notes | FileCheck %s

extern "C" void test_name1() {}
void test_name2() {}

// CHECK: !MDSubprogram(name: "test_name1",
// CHECK-NOT:           linkageName:
// CHECK-SAME:          ){{$}}
// CHECK: !MDSubprogram(name: "test_name2", linkageName: "_Z10test_name2v"
