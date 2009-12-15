// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s

int malloc(int a) { return a; }

