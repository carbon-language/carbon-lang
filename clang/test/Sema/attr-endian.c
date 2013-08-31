// RUN: %clang_cc1 %s -verify -fsyntax-only

int p1 __attribute__((endian(host)));
int p2 __attribute__((endian(device)));

int p3 __attribute__((endian));	// expected-error {{'endian' attribute requires parameter 1 to be an identifier}}
int p4 __attribute__((endian("host")));	// expected-error {{'endian' attribute requires parameter 1 to be an identifier}}
int p5 __attribute__((endian(host, 15)));	// expected-error {{'endian' attribute takes one argument}}
int p6 __attribute__((endian(strange)));	// expected-warning {{unknown endian 'strange'}}

void func(void) __attribute__((endian(host))); // expected-warning {{'endian' attribute only applies to variables}}
