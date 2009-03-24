// RUN: clang-cc %s -fsyntax-only -verify
// rdar://6587766

int fn1() __attribute__ ((warn_unused_result));
int fn2() __attribute__ ((pure));
int fn3() __attribute__ ((const));

int foo() {
  if (fn1() < 0 || fn2(2,1) < 0 || fn3(2) < 0)  // no warnings
    return -1;
  
  fn1();  // expected-warning {{expression result unused}}
  fn2(92, 21);  // expected-warning {{expression result unused}}
  fn3(42);  // expected-warning {{expression result unused}}
  return 0;
}

int bar __attribute__ ((warn_unused_result)); // expected-warning {{warning: 'warn_unused_result' attribute only applies to function types}}

