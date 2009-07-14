// RUN: clang-cc -emit-pch %S/resolve-loc-input.c -o %t.ast &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:1:8 | grep top_var &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:3:15 | grep top_func_decl &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:3:25 | grep param1 &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:5:17 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:5:23 | grep param2 &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:6:10 | grep local_var1 &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:7:15 | grep for_var &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:7:43 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:7:43 | grep '++for_var' &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:8:9 | grep local_var2 &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:8:30 | grep local_var2 &&
// RUN: index-test %t.ast -point-at %S/resolve-loc-input.c:8:30 | grep 'for_var + 1' &&
