// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code-aggressive -Wno-unused-value -Wno-covered-switch-default

// This previously triggered a warning from -Wunreachable-code because of
// a busted CFG.
typedef signed char BOOL;
BOOL radar10989084() {
  @autoreleasepool {  // no-warning
    return __objc_yes;
  }
}

// Test the warning works.
void test_unreachable() {
  return;
  return; // expected-warning {{will never be executed}}
}

#define NO __objc_no
#define YES __objc_yes
#define CONFIG NO

// Test that 'NO' and 'YES' are not treated as configuration macros.
int test_NO() {
  if (NO)
    return 1; // expected-warning {{will never be executed}}
  else
    return 0;
}

int test_YES() {
  if (YES)
    return 1;
  else
    return 0; // expected-warning {{will never be executed}}
}

int test_CONFIG() {
  if (CONFIG)
    return 1;
  else
    return 0;
}
