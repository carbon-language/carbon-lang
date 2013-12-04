// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

void test_eh_return_data_regno()
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 3
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 4
}
