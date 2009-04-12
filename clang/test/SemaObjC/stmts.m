// RUN: clang-cc %s -verify -fsyntax-only

// Note: NSException is not declared.
void f0(id x) {
  @try {
  } @catch (NSException *x) { // \
         expected-error{{unknown type name 'NSException'}}
  }
}

