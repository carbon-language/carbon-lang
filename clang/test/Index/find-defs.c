// RUN: clang-cc -fblocks -emit-pch %S/Inputs/t1.c -o %t1.ast
// RUN: clang-cc -fblocks -emit-pch %S/Inputs/t2.c -o %t2.ast

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:1:14 -print-defs > %t
// RUN: cat %t | count 1
// RUN: grep 't2.c:3:5,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:3:9 -print-defs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.c:3:6,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/foo.h:4:9 -print-defs > %t
// RUN: cat %t | count 1
// RUN: grep 't2.c:5:6,' %t

// RUN: index-test %t1.ast %t2.ast -point-at %S/Inputs/t1.c:8:7 -print-defs > %t
// RUN: cat %t | count 1
// RUN: grep 't2.c:5:6,' %t
