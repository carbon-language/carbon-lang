// RUN: clang-cc -fsyntax-only -verify -ffreestanding %s

int malloc(int a) { return a; }

