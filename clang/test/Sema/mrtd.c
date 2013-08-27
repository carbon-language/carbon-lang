// RUN: %clang_cc1 -DMRTD -mrtd -triple i386-unknown-unknown -verify %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -verify %s

#ifndef MRTD
// expected-note@+5 {{previous declaration is here}}
// expected-error@+5 {{function declared 'stdcall' here was previously declared without calling convention}}
// expected-note@+5 {{previous declaration is here}}
// expected-error@+5 {{function declared 'stdcall' here was previously declared without calling convention}}
#endif
void nonvariadic1(int a, int b, int c);
void __attribute__((stdcall)) nonvariadic1(int a, int b, int c);
void nonvariadic2(int a, int b, int c);
void __attribute__((stdcall)) nonvariadic2(int a, int b, int c) { }

// expected-note@+2 {{previous declaration is here}}
// expected-error@+2 {{function declared 'stdcall' here was previously declared without calling convention}}
void variadic(int a, ...);
void __attribute__((stdcall)) variadic(int a, ...);

#ifdef MRTD
// expected-note@+3 {{previous definition is here}}
// expected-error@+3 {{redefinition of 'a' with a different type: 'void ((*))(int, int) __attribute__((cdecl))' vs 'void (*)(int, int) __attribute__((stdcall))'}}
#endif
extern void (*a)(int, int);
__attribute__((cdecl)) extern void (*a)(int, int);

extern void (*b)(int, ...);
__attribute__((cdecl)) extern void (*b)(int, ...);

#ifndef MRTD
// expected-note@+3 {{previous definition is here}}
// expected-error@+3 {{redefinition of 'c' with a different type: 'void ((*))(int, int) __attribute__((stdcall))' vs 'void (*)(int, int)'}}
#endif
extern void (*c)(int, int);
__attribute__((stdcall)) extern void (*c)(int, int);

// expected-note@+2 {{previous definition is here}}
// expected-error@+2 {{redefinition of 'd' with a different type: 'void ((*))(int, ...) __attribute__((stdcall))' vs 'void (*)(int, ...)'}}
extern void (*d)(int, ...);
__attribute__((stdcall)) extern void (*d)(int, ...);
