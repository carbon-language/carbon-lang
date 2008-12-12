// RUN: clang %s -fsyntax-only -verify -pedantic

int a() {int p; *(1 ? &p : (void*)(0 && (a(),1))) = 10;}

// rdar://6091492 - ?: with __builtin_constant_p as the operand is an i-c-e.
int expr;
char w[__builtin_constant_p(expr) ? expr : 1];

