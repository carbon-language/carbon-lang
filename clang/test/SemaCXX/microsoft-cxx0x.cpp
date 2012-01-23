// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wc++11-narrowing -Wmicrosoft -verify -fms-extensions -std=c++11


struct A {
     unsigned int a;
};
int b = 3;
A var = {  b }; // expected-warning {{ cannot be narrowed }} expected-note {{override}}
