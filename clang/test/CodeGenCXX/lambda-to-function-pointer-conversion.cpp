// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc19.0.0 -std=c++11 -emit-llvm -o - %s | FileCheck %s

// This code used to cause an assertion failure in EmitDelegateCallArg.

// CHECK: define internal void @"?__invoke@<lambda_0>@?0??test@@YAXXZ@CA@UTrivial@@@Z"(
// CHECK: call void @"??R<lambda_0>@?0??test@@YAXXZ@QEBA@UTrivial@@@Z"(

// CHECK: define internal void @"??R<lambda_0>@?0??test@@YAXXZ@QEBA@UTrivial@@@Z"(

struct Trivial {
  int x;
};

void (*fnptr)(Trivial);

void test() {
  fnptr = [](Trivial a){ (void)a; };
}
