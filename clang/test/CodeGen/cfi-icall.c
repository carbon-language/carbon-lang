// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=MS %s

// Tests that we assign appropriate identifiers to unprototyped functions.

void f() {
}

void xf();

void g(int b) {
  void (*fp)() = b ? f : xf;
  // ITANIUM: call i1 @llvm.bitset.test(i8* {{.*}}, metadata !"_ZTSFvE")
  fp();
}

// ITANIUM-DAG: !{!"_ZTSFvE", void ()* @f, i64 0}
// ITANIUM-DAG: !{!"_ZTSFvE", void (...)* @xf, i64 0}
// MS-DAG: !{!"?6AX@Z", void ()* @f, i64 0}
// MS-DAG: !{!"?6AX@Z", void (...)* @xf, i64 0}
