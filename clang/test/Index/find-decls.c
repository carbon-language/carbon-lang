// RUN: %clang_cc1 -fblocks -emit-pch %S/Inputs/t1.c -o %t1.ast
// RUN: %clang_cc1 -fblocks -emit-pch %S/Inputs/t2.c -o %t2.ast

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:8:7 -print-decls > %t
// RUN: cat %t | count 3
// RUN: grep 'foo.h:4:6,' %t | count 2
// RUN: grep 't2.c:5:6,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:5:47 -print-decls > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:5:12,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:6:20 -print-decls > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:3:19,' %t

// field test

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:21:6 -print-decls > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:12:7,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:22:21 -print-decls > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:16:7,' %t
