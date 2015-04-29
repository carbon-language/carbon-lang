// RUN: %clang_cc1 -x c++ -g -emit-llvm -triple x86_64-linux-gnu -o - %s | FileCheck %s
// PR23332

// CHECK: DILocalVariable(tag: DW_TAG_arg_variable, arg: 255
// CHECK: DILocalVariable(tag: DW_TAG_arg_variable, arg: 256
// CHECK: DILocalVariable(tag: DW_TAG_arg_variable, arg: 257
void fn1(int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int, int, int, int, int, int, int, int, int, int,
         int, int, int, int, int) {}
