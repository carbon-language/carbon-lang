// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

// Tests that we assign unnamed metadata nodes to functions whose types have
// internal linkage.

namespace {

struct S {};

void f(S *s) {
}

}

void g() {
  void (*fp)(S *) = f;
  // CHECK: call i1 @llvm.bitset.test(i8* {{.*}}, metadata ![[VOIDS:[0-9]+]])
  fp(0);
}

// ITANIUM: !{![[VOIDS]], void (%"struct.(anonymous namespace)::S"*)* @_ZN12_GLOBAL__N_11fEPNS_1SE, i64 0}
// MS: !{![[VOIDS]], void (%"struct.(anonymous namespace)::S"*)* @"\01?f@?A@@YAXPEAUS@?A@@@Z", i64 0}
