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

// RUN: %clang_cc1 -emit-pch %s -o %t.ast
// RUN: c-index-test \
// RUN:   -cursor-at=%s:3:8 -cursor-at=%s:5:15 -cursor-at=%s:5:25 \
// RUN:   -cursor-at=%s:7:17 -cursor-at=%s:7:23 -cursor-at=%s:8:10 \
// RUN:   -cursor-at=%s:9:15 -cursor-at=%s:10:9 -cursor-at=%s:15:10 \
// RUN: %s | FileCheck %s
// CHECK: VarDecl=top_var
// CHECK: FunctionDecl=top_func_decl
// CHECK: ParmDecl=param1
// CHECK: FunctionDecl=top_func_def
// CHECK: ParmDecl=param2
// CHECK: VarDecl=local_var1
// CHECK: VarDecl=for_var
// CHECK: VarDecl=local_var2
// CHECK: FieldDecl=field_var

// FIXME: Eliminate these once clang_getCursor supports them.
// RUN: index-test %t.ast -point-at %s:9:43 > %t
// RUN: grep '++for_var' %t

// RUN: index-test %t.ast -point-at %s:10:30 > %t
// RUN: grep 'for_var + 1' %t
