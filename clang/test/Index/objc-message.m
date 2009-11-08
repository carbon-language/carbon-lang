// RUN: clang-cc -emit-pch %S/Inputs/t1.m -o %t1.m.ast
// RUN: clang-cc -emit-pch %S/Inputs/t2.m -o %t2.m.ast

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/objc.h:5:13 -print-refs > %t
// RUN: cat %t | count 1
// RUN: grep 't1.m:6:3,' %t

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/objc.h:6:13 -print-refs > %t
// RUN: cat %t | count 2
// RUN: grep 't1.m:7:3,' %t
// RUN: grep 't2.m:7:3,' %t

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/objc.h:10:13 -print-refs > %t
// RUN: cat %t | count 2
// RUN: grep 't1.m:6:3,' %t
// RUN: grep 't2.m:6:3,' %t

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/t1.m:6:15 -print-decls > %t
// RUN: cat %t | count 6
// RUN: grep 'objc.h:5:1,' %t | count 2
// RUN: grep 'objc.h:10:1,' %t | count 2
// RUN: grep 't1.m:15:1,' %t
// RUN: grep 't2.m:11:1,' %t

// RUN: index-test %t1.m.ast %t2.m.ast -point-at %S/Inputs/t1.m:7:15 -print-decls > %t
// RUN: cat %t | count 3
// RUN: grep 'objc.h:6:1,' %t | count 2
// RUN: grep 't1.m:18:1,' %t

// RUN: index-test %t2.m.ast %t1.m.ast -point-at %S/Inputs/t2.m:6:15 -print-decls > %t
// RUN: cat %t | count 3
// RUN: grep 'objc.h:10:1,' %t | count 2
// RUN: grep 't2.m:11:1,' %t

// RUN: index-test %t2.m.ast %t1.m.ast -point-at %S/Inputs/t2.m:7:15 -print-decls > %t
// RUN: cat %t | count 3
// RUN: grep 'objc.h:6:1,' %t | count 2
// RUN: grep 't1.m:18:1,' %t
