// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -ast-print %s
// RUN: %clang_cc1 -ast-dump %s
// FIXME: %clang_cc1 -ast-print-xml -o %t %s
// RUN: %clang_cc1 -print-decl-contexts %s
// RUN: %clang_cc1 -dump-record-layouts %s

#include "cxx-language-features.inc"
