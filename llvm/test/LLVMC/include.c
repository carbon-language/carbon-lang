/*
 * Check that the 'include' options work.
 * RUN: echo "int x;\n" > %t1.inc
 * RUN: llvmc2 -include %t1.inc -fsyntax-only %s
 */

int f0(void) {
  return x;
}
