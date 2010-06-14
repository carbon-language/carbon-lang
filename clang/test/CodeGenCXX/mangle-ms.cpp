// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-apple-darwin10 | FileCheck %s

// CHECK: @"\01?a@@3HA"
// CHECK: @"\01?b@N@@3HA"
// CHECK: @c
// CHECK: @"\01?d@foo@@0FB"
// CHECK: @"\01?e@foo@@1JC"
// CHECK: @"\01?f@foo@@2DD"

int a;

namespace N { int b; }

static int c;
int _c(void) {return c;}

class foo {
  static const short d;
protected:
  static volatile long e;
public:
  static const volatile char f;
};

const short foo::d = 0;
volatile long foo::e;
const volatile char foo::f = 'C';

