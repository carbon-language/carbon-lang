// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify %s

int __attribute__((target("avx,sse4.2,arch=ivybridge"))) foo() { return 4; }
int __attribute__((target())) bar() { return 4; } //expected-error {{'target' attribute takes one argument}}


