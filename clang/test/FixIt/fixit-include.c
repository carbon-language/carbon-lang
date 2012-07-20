// RUN: %clang_cc1 -fsyntax-only -Wall -pedantic -verify %s
// RUN: cp %s %t
// RUN: cp %S/fixit-include.h %T
// RUN: not %clang_cc1 -fsyntax-only -fixit %t
// RUN: %clang_cc1 -Wall -pedantic %t

#include <fixit-include.h> // expected-error {{'fixit-include.h' file not found with <angled> include; use "quotes" instead}}

#pragma does_not_exist // expected-warning {{unknown pragma ignored}}

int main( void ) {
  return 0;
}
