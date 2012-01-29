// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s

void test0() __attribute__((visibility("default")));
void test1() __attribute__((visibility("hidden")));
void test2() __attribute__((visibility("internal")));

// rdar://problem/10753392
void test3() __attribute__((visibility("protected"))); // expected-warning {{target does not support 'protected' visibility; using 'default'}}

