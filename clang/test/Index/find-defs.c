// RUN: clang-cc -emit-pch %S/t1.c -o %t1.ast &&
// RUN: clang-cc -emit-pch %S/t2.c -o %t2.ast &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:1:14 -print-defs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:1:14 -print-defs | grep 't2.c:3:5,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:3:9 -print-defs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:3:9 -print-defs | grep 't1.c:3:6,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:4:9 -print-defs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/foo.h:4:9 -print-defs | grep 't2.c:5:6,' &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:8:7 -print-defs | count 1 &&
// RUN: index-test %t1.ast %t2.ast -point-at %S/t1.c:8:7 -print-defs | grep 't2.c:5:6,'
