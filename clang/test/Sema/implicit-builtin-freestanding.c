// RUN: clang -fsyntax-only -verify -ffreestanding %s

int malloc(int a) { return a; }

