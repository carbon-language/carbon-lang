// RUN: clang -parse-ast -verify %s

void f (int p[]) { p++; }

