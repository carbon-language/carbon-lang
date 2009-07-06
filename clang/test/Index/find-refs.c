// RUN: clang-cc -emit-pch %S/t1.c -o %t1.ast &&
// RUN: clang-cc -emit-pch %S/t2.c -o %t2.ast &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:1:14 -print-refs | count 3 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:1:14 -print-refs | grep 't1.c:4:19,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:1:14 -print-refs | grep 't2.c:6:3,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:1:14 -print-refs | grep 't2.c:7:12,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:3:9 -print-refs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:3:9 -print-refs | grep 't2.c:7:3,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:4:9 -print-refs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:4:9 -print-refs | grep 't1.c:8:3,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:3:22 -print-refs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:3:22 -print-refs | grep 't1.c:6:17,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:4:11 -print-refs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:4:11 -print-refs | grep 't1.c:6:5,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:5:30 -print-refs | count 3 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:5:30 -print-refs | grep 't1.c:5:27,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:5:30 -print-refs | grep 't1.c:5:44,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:5:30 -print-refs | grep 't1.c:6:26,'
