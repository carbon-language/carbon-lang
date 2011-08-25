// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t1 %S/Inputs/chain-trivial1.h
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t2 -include-pch %t1 %S/Inputs/chain-trivial2.h
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-print -include-pch %t2 %s | FileCheck %s
// CHECK: struct __va_list_tag {
