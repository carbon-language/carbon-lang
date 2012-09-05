// RUN: %clang_cc1 -fsyntax-only -fno-wchar -verify %s
wchar_t x; // expected-error {{unknown type name 'wchar_t'}}
