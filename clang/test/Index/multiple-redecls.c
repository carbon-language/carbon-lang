// RUN: %clang_cc1 -emit-pch %s -o %t.ast
// RUN: index-test %t.ast -point-at %s:8:4 -print-decls | count 2
// RUN: index-test %t.ast -point-at %s:8:4 -print-defs | count 1

static void foo(int x);

static void bar(void) {
  foo(10);
}

void foo(int x) { 
}
