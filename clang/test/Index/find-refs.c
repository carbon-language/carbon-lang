// RUN: clang-cc -fblocks -emit-pch %S/Inputs/t1.c -o %t1.ast
// RUN: clang-cc -fblocks -emit-pch %S/Inputs/t2.c -o %t2.ast

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:1:14 -print-refs > %t
// RUN: cat %t | count 4
// RUN: grep 't1.c:4:19,' %t
// RUN: grep 't1.c:28:40,' %t
// RUN: grep 't2.c:6:3,' %t
// RUN: grep 't2.c:7:12,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:3:9 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't2.c:7:3,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:4:9 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:8:3,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:3:22 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:6:17,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:4:11 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:6:5,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:5:30 -print-refs > %t
// RUN: cat %t | count 3
// RUN: grep 't1.c:5:27,' %t
// RUN: grep 't1.c:5:44,' %t
// RUN: grep 't1.c:6:26,' %t

// field test

// FIXME: References point at the start of MemberExpr, make them point at the field instead.
// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:12:7 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:21:3,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:16:7 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:22:3,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:7:11 -print-refs > %t
// RUN: cat %t | count 2
// RUN: grep 't1.c:25:3,' %t
// RUN: grep 't2.c:10:3,' %t
