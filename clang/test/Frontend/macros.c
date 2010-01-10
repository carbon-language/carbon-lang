// RUN: %clang_cc1 -DA= -DB=1 -verify -fsyntax-only %s

int a[(B A) == 1 ? 1 : -1];

