// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://8360841

wchar_t s[] = L"\u2722";

// CHECK: @s = global [8 x i8] c"\22'\00\00\00\00\00\00"
