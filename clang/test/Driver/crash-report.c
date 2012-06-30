// RUN: %clang -fsyntax-only %s 2>&1 | FileCheck %s
// REQUIRES: crash-recovery
// XFAIL: mingw32,win32

#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK-NEXT: {{.*}}: note: diagnostic msg: {{.*}}.c
