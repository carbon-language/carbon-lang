// RUN: %clang -fexceptions -S -emit-llvm %s -o /dev/null -pedantic-errors
// rdar://6551276

void foo(const unsigned short *);
void bar(void) {
  unsigned short *s[3];
  int i;
  @try { } @catch (id anException) { }
  foo(2+s[i]);
}
