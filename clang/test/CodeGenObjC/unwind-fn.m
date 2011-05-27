// RUN: %clang_cc1 -fobjc-nonfragile-abi -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck --check-prefix=DEFAULT_EH %s
// RUN: %clang_cc1 -fsjlj-exceptions -fobjc-nonfragile-abi -fexceptions -fobjc-exceptions -emit-llvm -o - %s | FileCheck --check-prefix=SJLJ_EH %s

// DEFAULT_EH: declare void @_Unwind_Resume(i8*)
// SJLJ_EH: declare void @_Unwind_SjLj_Resume(i8*)

void f1(), f2();
void f0() {
  @try {
    f1();
  } @catch (...) {
    f2();
  }
}
