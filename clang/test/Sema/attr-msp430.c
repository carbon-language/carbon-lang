// RUN: %clang_cc1 -triple msp430-unknown-unknown -fsyntax-only -verify %s

int i;
void f(void) __attribute__((interrupt(i))); /* expected-error {{'interrupt' attribute requires an integer constant}} */

void f2(void) __attribute__((interrupt(12)));
