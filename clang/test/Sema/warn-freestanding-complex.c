// RUN: clang-cc -fsyntax-only -ffreestanding -pedantic -verify %s

void foo(float _Complex c) { // expected-warning{{complex numbers are an extension in a freestanding C99 implementation}}
}
