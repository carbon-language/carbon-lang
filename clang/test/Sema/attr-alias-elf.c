// RUN: %clang_cc1 -triple x86_64-pc-linux  -fsyntax-only -verify -emit-llvm-only %s

void f1(void) __attribute__((alias("g1")));
void g1(void) {
}

void f2(void) __attribute__((alias("g2"))); // expected-error {{alias must point to a defined variable or function}}


void f3(void) __attribute__((alias("g3"))); // expected-error {{alias must point to a defined variable or function}}
void g3(void);

extern int a1 __attribute__((alias("b1")));
int b1 = 42;

extern int a2 __attribute__((alias("b2"))); // expected-error {{alias must point to a defined variable or function}}

extern int a3 __attribute__((alias("b3"))); // expected-error {{alias must point to a defined variable or function}}
extern int b3;

extern int a4 __attribute__((alias("b4"))); // expected-error {{alias must point to a defined variable or function}}
typedef int b4;
