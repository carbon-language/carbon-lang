// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wc++0x-narrowing -Wmicrosoft -verify -fms-extensions -std=c++0x


struct A {
     unsigned int a;
};
int b = 3;
A var = {  b }; // expected-warning {{ cannot be narrowed }} expected-note {{override}}


