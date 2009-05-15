// RUN: clang-cc -fsyntax-only -verify %s

int i;

void foo() {
  (throw,throw);
  (1 ? throw 1 : throw 2);
  throw int(1);
  throw;
  throw 1;
  throw;
  1 ? throw : (void)42;
  // gcc doesn't parse the below, but we do
  __extension__ throw 1;
  (void)throw;              // expected-error {{expected expression}}
}
