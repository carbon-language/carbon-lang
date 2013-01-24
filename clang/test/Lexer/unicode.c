// RUN: %clang_cc1 -fsyntax-only -verify %s

// This file contains Unicode characters; please do not "fix" them!

extern int x; // expected-warning {{treating Unicode character as whitespace}}
extern int　x; // expected-warning {{treating Unicode character as whitespace}}
