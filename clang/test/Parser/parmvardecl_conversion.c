// RUN: clang -parse-ast-check %s

void f (int p[]) { p++; }

