// RUN: %clang_cc1 -triple x86_64-windows -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify -emit-llvm-only -DCHECK_ALIASES %s
// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify -emit-llvm-only %s

#if defined(_WIN32)
void foo(void) {}
void bar(void) __attribute__((ifunc("foo")));
// expected-warning@-1 {{unknown attribute 'ifunc' ignored}}

#else
#if defined(CHECK_ALIASES)
void *f1_ifunc(void);
void f1(void) __attribute__((ifunc("f1_ifunc")));
// expected-error@-1 {{ifunc must point to a defined function}}

void *f2_a(void) __attribute__((alias("f2_b")));
void *f2_b(void) __attribute__((ifunc("f2_a")));
// expected-error@-1 {{ifunc definition is part of a cycle}}

void *f3_a(void) __attribute__((ifunc("f3_b")));
// expected-warning@-1 {{ifunc will always resolve to f3_c even if weak definition of f3_b is overridden}}
void *f3_b(void) __attribute__((weak, alias("f3_c")));
void *f3_c(void) { return 0; }

void f4_ifunc(void) {}
void f4(void) __attribute__((ifunc("f4_ifunc")));
// expected-error@-1 {{ifunc resolver function must return a pointer}}

int f5_resolver_gvar;
void f5(void) __attribute__((ifunc("f5_resolver_gvar")));
// expected-error@-1 {{ifunc must point to a defined function}}

void *f6_resolver_resolver(void) { return 0; }
void *f6_resolver(void) __attribute__((ifunc("f6_resolver_resolver")));
void f6(void) __attribute__((ifunc("f6_resolver")));
// expected-error@-1 {{ifunc must point to a defined function}}

#else
void f1a(void) __asm("f1");
void f1a(void) {}
// expected-note@-1 {{previous definition is here}}
void f1(void) __attribute__((ifunc("f1_ifunc")));
// expected-error@-1 {{definition with same mangled name 'f1' as another definition}}
void *f1_ifunc(void) { return 0; }

void *f6_ifunc(int i);
void __attribute__((ifunc("f6_ifunc"))) f6(void) {}
// expected-error@-1 {{definition 'f6' cannot also be an ifunc}}

#endif
#endif
