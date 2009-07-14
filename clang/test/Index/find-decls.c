// RUN: clang-cc -emit-pch %S/t1.c -o %t1.ast &&
// RUN: clang-cc -emit-pch %S/t2.c -o %t2.ast &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:8:7 -print-decls | count 3 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:8:7 -print-decls | grep 'foo.h:4:6,' | count 2 && 
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:8:7 -print-decls | grep 't2.c:5:6,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:5:47 -print-decls | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:5:47 -print-decls | grep 't1.c:5:12,' && 
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:6:20 -print-decls | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:6:20 -print-decls | grep 't1.c:3:19,' &&

// field test
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:21:6 -print-decls | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:21:6 -print-decls | grep 't1.c:12:7,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:22:21 -print-decls | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:22:21 -print-decls | grep 't1.c:16:7,'
