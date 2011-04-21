// RUN: c-index-test -test-load-source-memory-usage none %s 2>&1 | FileCheck %s

// rdar://9275920 - We would create millions of Exprs to fill out the initializer.

double data[1000000] = {0};

struct S {
 S(int);
 S();
};

S data2[1000000] = {0};

// CHECK: TOTAL = {{.*}} (0.{{.*}} MBytes)
