// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null

int foo() {
#ifdef __ppc__
  register int X __asm__("r1");
#else
  register int X __asm__("ebx");
#endif
  return X;
}
