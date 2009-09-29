// Run lines are sensitive to line numbers and come below the code.

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
// RUN: index-test %t.ast -point-at %s:3:8 | grep top_var &&
// RUN: index-test %t.ast -point-at %s:5:15 | grep top_func_decl &&
// RUN: index-test %t.ast -point-at %s:5:25 | grep param1 &&
// RUN: index-test %t.ast -point-at %s:7:17 | grep top_func_def &&
// RUN: index-test %t.ast -point-at %s:7:23 | grep param2 &&
// RUN: index-test %t.ast -point-at %s:8:10 | grep local_var1 &&
// RUN: index-test %t.ast -point-at %s:9:15 | grep for_var &&

// RUN: index-test %t.ast -point-at %s:9:43 > %t &&
// RUN: grep '++for_var' %t &&

// RUN: index-test %t.ast -point-at %s:10:9 | grep local_var2 &&

// RUN: index-test %t.ast -point-at %s:10:30 > %t &&
// RUN: grep 'for_var + 1' %t &&

// fields test.
// RUN: index-test %t.ast -point-at %s:15:10 | grep field_var
