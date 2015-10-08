// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o %t %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-bc -o %t %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -debug-info-kind=limited -emit-llvm-bc -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm-bc -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -debug-info-kind=limited -emit-llvm-bc -o %t %s

#include "c-language-features.inc"
