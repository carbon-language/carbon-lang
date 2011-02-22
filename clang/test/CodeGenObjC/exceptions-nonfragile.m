// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-nonfragile-abi -fexceptions -fobjc-exceptions -O2 -o - %s | FileCheck %s

// rdar://problem/8535238
// CHECK: declare void @objc_exception_rethrow()

void protos() {
  extern void foo();
  @try {
    foo();
  } @catch (id e) {
    @throw;
  }
}

void throwing() {
  @throw(@"error!");
}
