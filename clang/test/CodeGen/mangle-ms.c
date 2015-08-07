// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s

// CHECK: define void @"\01?f@@$$J0YAXP6AX@Z@Z"
__attribute__((overloadable)) void f(void (*x)()) {}
