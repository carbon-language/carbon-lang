/* RUN: clang -parse-ast %s -std=c89 &&
 * RUN: not clang -parse-ast %s -std=c99 -pedantic-errors
 */

int A() {
  return X();
}

