// RUN: not --crash %clang_cc1 -analyze -analyzer-checker=debug.ExprInspection %s 2>&1 | FileCheck %s
// REQUIRES: crash-recovery

void clang_analyzer_crash(void);

void inlined() {
  clang_analyzer_crash();
}

void test() {
  inlined();
}

// CHECK: 0.	Program arguments: {{.*}}clang
// CHECK-NEXT: 1.	<eof> parser at end of file
// CHECK-NEXT: 2. While analyzing stack: 
// CHECK-NEXT:  #0 void inlined()
// CHECK-NEXT:  #1 void test()
// CHECK-NEXT: 3.	{{.*}}crash-trace.c:{{[0-9]+}}:3: Error evaluating statement
