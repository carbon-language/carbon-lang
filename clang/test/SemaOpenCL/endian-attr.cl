// RUN: %clang_cc1 -verify %s

constant long a __attribute__((endian(host))) = 100;

constant long b __attribute__((endian(device))) = 100;

constant long c __attribute__((endian(none))) = 100; // expected-warning {{unknown endian 'none'}}

void func() __attribute__((endian(host))); // expected-warning {{endian attribute only applies to variables}}
