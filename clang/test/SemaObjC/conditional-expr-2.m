// RUN: clang -fsyntax-only -verify %s

@interface A
@end
@interface B
@end

void f0(int cond, A *a, B *b) {
  // Ensure that we can still send a message to result of incompatible
  // conditional expression.
  [ (cond ? a : b) test ]; // expected-warning {{method '-test' not found}}
}
