/* RUN: clang -fsyntax-only %s -std=c89 &&
 * RUN: not clang -fsyntax-only %s -std=c99 -pedantic-errors
 */

int A() {
  return X();
}

