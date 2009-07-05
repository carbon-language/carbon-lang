// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:8:7 -print-decls | count 3 &&
// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:8:7 -print-decls | grep 'foo.h:4:6,' | count 2 && 
// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:8:7 -print-decls | grep 't2.c:5:6,' &&
// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:5:47 -print-decls | count 1 &&
// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:5:47 -print-decls | grep 't1.c:5:12,' && 
// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:6:20 -print-decls | count 1 &&
// RUN: index-test %T/t1.ast %T/t2.ast -point-at %S/t1.c:6:20 -print-decls | grep 't1.c:3:19,'
