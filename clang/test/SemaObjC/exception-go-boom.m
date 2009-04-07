// RUN: clang-cc %s -verify -fsyntax-only

// Note: NSException is not declared.
void f0(id x) {
  @try {
  } @catch (NSException *x) { // expected-warning{{type specifier missing, defaults to 'int'}} expected-error{{@catch parameter is not an Objective-C class type}}
  }
}

