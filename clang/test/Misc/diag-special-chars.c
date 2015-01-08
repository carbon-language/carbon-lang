// RUN: %clang_cc1 %s -verify
// RUN: not %clang_cc1 %s 2>&1 | FileCheck %s

// There are two special characters on the following line, one which is used
// as a marker character for diagnostic printing.  Ensure diagnostics involving
// these characters do not cause problems with the diagnostic printer.
#error Hi   Bye
//expected-error@-1 {{Hi   Bye}}

// CHECK: error: Hi   Bye
// CHECK: #error Hi <U+007F> <U+0080> Bye
