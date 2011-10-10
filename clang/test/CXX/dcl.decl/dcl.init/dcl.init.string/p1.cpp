// RUN: %clang_cc1 -fsyntax-only -verify %s

char x1[]("hello");
extern char x1[6];

char x2[] = "hello";
extern char x2[6];
