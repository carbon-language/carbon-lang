// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -o - -test-coverage -femit-coverage-notes | FileCheck %s

extern "C" void test_name1() {}
void test_name2() {}

// CHECK: metadata !"0x2e\00test_name1\00test_name1\00\00{{[^,]+}}", {{.*}} DW_TAG_subprogram
// CHECK: metadata !"0x2e\00test_name2\00test_name2\00_Z10test_name2v\00{{[^,]+}}", {{.*}} DW_TAG_subprogram
