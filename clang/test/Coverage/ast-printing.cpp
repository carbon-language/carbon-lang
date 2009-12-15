// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -ast-print %s
// RUN: %clang_cc1 -ast-dump %s
// FIXME: %clang_cc1 -ast-print-xml -o %t %s

#include "cxx-language-features.inc"
