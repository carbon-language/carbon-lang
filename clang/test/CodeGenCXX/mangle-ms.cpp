// RUN: %clang_cc1 -fms-extensions -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-apple-darwin10 | FileCheck %s

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
// CHECK: @"\01?_c@@YAHXZ"

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

// Static functions are mangled, too.
// Also make sure calling conventions, arglists, and throw specs work.
static void __stdcall alpha(float a, double b) throw() {}
bool __fastcall beta(long long a, wchar_t b) throw(signed char, unsigned char) {
// CHECK: @"\01?beta@@YI_N_J_W@CE@"
  alpha(0.f, 0.0);
  return false;
}

// CHECK: @"\01?alpha@@YGXMN@@"

