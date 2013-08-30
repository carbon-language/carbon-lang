// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 -emit-llvm < %s | FileCheck -check-prefix=FREEBSD %s
// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm < %s | FileCheck -check-prefix=WIN64 %s

void __attribute__((ms_abi)) f1(void);
void __attribute__((sysv_abi)) f2(void);
void f3(void) {
// FREEBSD: define void @f3()
// WIN64: define void @f3()
  f1();
// FREEBSD: call x86_64_win64cc void @f1()
// WIN64: call void @f1()
  f2();
// FREEBSD: call void @f2()
// WIN64: call x86_64_sysvcc void @f2()
}
// FREEBSD: declare x86_64_win64cc void @f1()
// FREEBSD: declare void @f2()
// WIN64: declare void @f1()
// WIN64: declare x86_64_sysvcc void @f2()

