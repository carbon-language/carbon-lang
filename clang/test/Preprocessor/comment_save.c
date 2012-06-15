// RUN: %clang_cc1 -E -C %s | FileCheck -strict-whitespace %s

// foo
// CHECK: // foo

/* bar */
// CHECK: /* bar */

#if FOO
#endif
/* baz */
// CHECK: /* baz */
