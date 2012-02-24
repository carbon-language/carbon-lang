// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s -Wgnu
extern int x;
__decltype(1) x = 3; // expected-warning {{is a GNU extension}}
