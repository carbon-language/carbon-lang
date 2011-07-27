// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null
// XFAIL: *
// XTARGET: arm, x86, x86_64

int foo() {
#ifdef __arm__
  register int X __asm__("r1");
#else
  register int X __asm__("ebx");
#endif
  return X;
}
