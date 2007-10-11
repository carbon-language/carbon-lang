// RUN: clang -fsyntax-only -verify %s

void f1(int a) {
    if (a); // expected-warning {{if statement has empty body}}
}

void f2(int a) {
    if (a) {}
}
