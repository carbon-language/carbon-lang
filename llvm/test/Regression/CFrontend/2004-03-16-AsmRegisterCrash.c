int foo() {
  register int X __asm__("ebx");
  return X;
}
