// RUN: %clang_cc1 %s -triple=arm-linux-gnueabi -target-abi aapcs -emit-llvm -o - | FileCheck %s

class SMLoc {
 const char *Ptr;
public:
 SMLoc();
 SMLoc(const SMLoc &RHS);
};
SMLoc foo(void *p);
void bar(void *x) {
 foo(x);
}
void zed(SMLoc x);
void baz() {
  SMLoc a;
  zed(a);
}

// CHECK: declare arm_aapcscc void @_Z3fooPv(%class.SMLoc* sret, i8*)
// CHECK: declare arm_aapcscc void @_Z3zed5SMLoc(%class.SMLoc*)
