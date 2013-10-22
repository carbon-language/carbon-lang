// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

void f0() {}
void fun0(void) __attribute((alias("f0")));

void f1() {}
void fun1() {} // expected-note {{previous definition}}
void fun1(void) __attribute((alias("f1"))); // expected-error {{redefinition of 'fun1'}}

void f2() {}
void fun2(void) __attribute((alias("f2"))); // expected-note {{previous definition}}
void fun2() {} // expected-error {{redefinition of 'fun2'}}

void f3() {}
void fun3(void) __attribute((alias("f3"))); // expected-note {{previous definition}}
void fun3(void) __attribute((alias("f3"))); // expected-error {{redefinition of 'fun3'}}

void f4() {}
void fun4(void) __attribute((alias("f4")));
void fun4(void);

// FIXME: We should produce a special case error for this.
void f5() {}
void __attribute((alias("f5"))) fun5(void) {} // expected-error {{redefinition of 'fun5'}} // expected-note {{previous definition}}

int v1;
int var1 __attribute((alias("v1"))); // expected-note {{previous definition}}
int var1 __attribute((alias("v1"))); // expected-error {{redefinition of 'var1'}}

int v2;
int var2 = 2; // expected-note {{previous definition}}
int var2 __attribute((alias("v2"))); // expected-error {{redefinition of 'var2'}}

int v3;
int var3 __attribute((alias("v3"))); // expected-note {{previous definition}}
int var3 = 2; // expected-error {{redefinition of 'var3'}}

int v4;
int var4; // expected-note {{previous definition}}
int var4 __attribute((alias("v4"))); // expected-error {{alias definition of 'var4' after tentative definition}}

int v5;
int var5 __attribute((alias("v5"))); // expected-note {{previous definition}}
int var5; // expected-error {{tentative definition of 'var5' after alias definition}}
