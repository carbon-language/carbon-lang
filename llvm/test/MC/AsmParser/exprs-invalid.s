// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t
// RUN: FileCheck -input-file %t %s

// Currently XFAIL'ed, since the front-end isn't validating this. Figure out the
// right resolution.
//
// XFAIL: *

        .text
a:
        .data
// CHECK: expected relocatable expression
        .long -(0 + a)
