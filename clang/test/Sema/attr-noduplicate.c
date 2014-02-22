// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((noduplicate)); // expected-warning {{'noduplicate' attribute only applies to functions}}

void t1() __attribute__((noduplicate));

void t2() __attribute__((noduplicate(2))); // expected-error {{'noduplicate' attribute takes no arguments}}

