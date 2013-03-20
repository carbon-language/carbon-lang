// RUN: %clang_cc1 %s -emit-llvm -o - -test-coverage -femit-coverage-notes | FileCheck %s

extern "C" void test_name1() {}
void test_name2() {}

// CHECK: metadata !"test_name1", metadata !"test_name1", metadata !"",{{.*}}DW_TAG_subprogram
// CHECK: metadata !"test_name2", metadata !"test_name2", metadata !"_Z10test_name2v",{{.*}}DW_TAG_subprogram
