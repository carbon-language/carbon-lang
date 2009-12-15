// RUN: %clang_cc1 -fsyntax-only -verify %s
typedef struct {
        int a; int : 24; char b;
} S;

S a = { 1, 2 };
