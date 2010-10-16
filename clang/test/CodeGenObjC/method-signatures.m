// Check method signatures for synthesized runtime functions.
//
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 \
// RUN:    -fexceptions -fobjc-nonfragile-abi2 -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

// CHECK: declare void @objc_exception_rethrow()
void f1(void);
void f0() {
  @try {
    f1();
  } @finally {
  }
}
