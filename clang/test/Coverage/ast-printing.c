// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -ast-print %s
// RUN: %clang_cc1 -ast-dump %s
// RUN: %clang_cc1 -ast-print-xml -o %t %s

#include "c-language-features.inc"
