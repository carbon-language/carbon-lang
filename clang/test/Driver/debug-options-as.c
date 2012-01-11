// Check to make sure clang with -g on a .s file gets passed.
// rdar://9275556
// RUN: touch %t.s
// RUN: %clang -### -c -g %t.s 2>&1 | FileCheck -check-prefix=S %s
//
// cygming have not supported integrated-as yet.
// XFAIL: cygwin,mingw32
//
// S: "-cc1as"
// S: "-g"
