// RUN: touch %t
// RUN: chmod 0 %t
// RUN: not %clang_cc1 -E -dependency-file bla -MT %t -MP -o %t -x c /dev/null 2>&1 | FileCheck %s
// RUN: rm -f %t

// CHECK: error: unable to open output file

// rdar://9286457
