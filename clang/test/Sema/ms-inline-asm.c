// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fms-extensions -fenable-experimental-ms-inline-asm -verify -fsyntax-only

void t1(void) { 
 __asm __asm // expected-warning {{MS-style inline assembly is not supported}} expected-error {{__asm used with no assembly instructions}}
}
