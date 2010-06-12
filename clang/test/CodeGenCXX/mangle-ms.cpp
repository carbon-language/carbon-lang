// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-apple-darwin10 | FileCheck %s

// CHECK: @"\01?a@@"
// CHECK: @"\01?b@N@@"
// CHECK: @c

int a;

namespace N { int b; }

static int c;
int _c(void) {return c;}

