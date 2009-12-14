// RUN: clang -cc1 %s -verify -fsyntax-only

struct some_struct;

// Note: NSException is not declared.
void f0(id x) {
  @try {
  } @catch (NSException *x) { // expected-error {{unknown type name 'NSException'}}
  } @catch (struct some_struct x) { // expected-error {{@catch parameter is not a pointer to an interface type}}
  } @catch (int x) { // expected-error {{@catch parameter is not a pointer to an interface type}}
  } @catch (...) {
  }
}

