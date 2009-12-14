// RUN: clang -cc1 -emit-pch %S/Inputs/t1.m -o %t1.m.ast
// RUN: clang -cc1 -emit-pch %S/Inputs/t2.m -o %t2.m.ast

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/t1.m:12:12 -print-decls > %t
// RUN: cat %t | count 2
// RUN: grep 'objc.h:2:9,' %t | count 2

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/objc.h:5:13 -print-decls > %t
// RUN: cat %t | count 3
// RUN: grep 'objc.h:5:1,' %t | count 2
// RUN: grep 't1.m:15:1,' %t | count 1

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/objc.h:10:13 -print-decls > %t
// RUN: cat %t | count 3
// RUN: grep 'objc.h:10:1,' %t | count 2
// RUN: grep 't2.m:11:1,' %t | count 1
