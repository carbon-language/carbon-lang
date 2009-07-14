int top_var;

void top_func_decl(int param1);

void top_func_def(int param2) {
  int local_var1;
  for (int for_var = 100; for_var < 500; ++for_var) {
    int local_var2 = for_var + 1;
  }
}

struct S {
  int field_var;
};


// RUN: clang-cc -emit-pch %s -o %t.ast &&
// RUN: index-test %t.ast -point-at %s:1:8 | grep top_var &&
// RUN: index-test %t.ast -point-at %s:3:15 | grep top_func_decl &&
// RUN: index-test %t.ast -point-at %s:3:25 | grep param1 &&
// RUN: index-test %t.ast -point-at %s:5:17 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %s:5:23 | grep param2 &&
// RUN: index-test %t.ast -point-at %s:6:10 | grep local_var1 &&
// RUN: index-test %t.ast -point-at %s:7:15 | grep for_var &&
// RUN: index-test %t.ast -point-at %s:7:43 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %s:7:43 | grep '++for_var' &&
// RUN: index-test %t.ast -point-at %s:8:9 | grep local_var2 &&
// RUN: index-test %t.ast -point-at %s:8:30 | grep local_var2 &&
// RUN: index-test %t.ast -point-at %s:8:30 | grep 'for_var + 1' &&

// fields test.
// RUN: index-test %t.ast -point-at %s:13:10 | grep field_var
