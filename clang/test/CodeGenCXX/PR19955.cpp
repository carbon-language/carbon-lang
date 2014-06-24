// RUN: %clang_cc1 -triple i686-windows-msvc -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fno-rtti -emit-llvm -std=c++1y -O0 -o - %s | FileCheck %s

extern int __declspec(dllimport) x;
extern long long y;
// CHECK-DAG: @"\01?y@@3_JA" = global i64 0
long long y = (long long)&x;

// CHECK-LABEL: @"\01??__Ey@@YAXXZ"
// CHECK-DAG: @"\01?y@@3_JA"
