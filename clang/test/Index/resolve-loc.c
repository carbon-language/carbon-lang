// RUN: clang-cc -emit-pch %s -o %t.ast &&
// RUN: index-test %t.ast -point-at %s:15:8 | grep top_var &&
// RUN: index-test %t.ast -point-at %s:17:15 | grep top_func_decl &&
// RUN: index-test %t.ast -point-at %s:17:25 | grep param1 &&
// RUN: index-test %t.ast -point-at %s:19:17 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %s:19:23 | grep param2 &&
// RUN: index-test %t.ast -point-at %s:20:10 | grep local_var1 &&
// RUN: index-test %t.ast -point-at %s:21:15 | grep for_var &&
// RUN: index-test %t.ast -point-at %s:21:43 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %s:21:43 | grep '++for_var' &&
// RUN: index-test %t.ast -point-at %s:22:9 | grep local_var2 &&
// RUN: index-test %t.ast -point-at %s:22:30 | grep local_var2 &&
// RUN: index-test %t.ast -point-at %s:22:30 | grep 'for_var + 1'

int top_var;

void top_func_decl(int param1);

void top_func_def(int param2) {
  int local_var1;
  for (int for_var = 100; for_var < 500; ++for_var) {
    int local_var2 = for_var + 1;
  }
}
