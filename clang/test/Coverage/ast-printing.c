// RUN: clang-cc --fsyntax-only %s &&
// RUN: clang-cc --ast-print %s &&
// RUN: clang-cc --ast-dump %s &&
// RUN: clang-cc --ast-print-xml -o %t %s

#include "c-language-features.inc"
