// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -g -emit-llvm -o - | FileCheck %s

struct T {
  int method();
};

void foo(int (T::*method)()) {}

// A pointer to a member function is a pair of function- and this-pointer.
// CHECK: [ DW_TAG_ptr_to_member_type ] {{.*}} size 128
