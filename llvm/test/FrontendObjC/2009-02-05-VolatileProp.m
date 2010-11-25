/* RUN: %llvmgcc -w -x objective-c -S %s -o /dev/null -pedantic-errors
   rdar://6551276 */

void foo(const unsigned short *);
void bar() {
  unsigned short *s[3];
  int i;
  @try { } @catch (id anException) { }
  foo(2+s[i]);
}

