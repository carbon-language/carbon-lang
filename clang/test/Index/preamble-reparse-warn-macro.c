// RUN: mkdir -p %t
// RUN: touch %t/header.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 2 local %s -I %t 2> %t.err.txt > %t.out.txt
// RUN: cat %t.err.txt >> %t.out.txt
// RUN: FileCheck -input-file=%t.out.txt %s
// CHECK: preamble-reparse-warn-macro.c:[[@LINE+8]]:9: warning: 'MAC' macro redefined
// CHECK-NEXT: Number FIX-ITs = 0
// CHECK-NEXT: preamble-reparse-warn-macro.c:[[@LINE+2]]:9: note: previous definition is here

#define MAC 1
#include "header.h"

void test();
#define MAC 2
