// RUN: clang-cc -E -C %s | FileCheck -strict-whitespace %s

// foo
// CHECK: // foo

/* bar */
// CHECK: /* bar */

