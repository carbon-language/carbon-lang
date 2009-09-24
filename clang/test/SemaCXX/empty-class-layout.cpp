// RUN: clang-cc -triple x86_64-unknown-unknown %s -fsyntax-only -verify 

#define SA(n, p) int a##n[(p) ? 1 : -1]

struct A { int a; };
SA(0, sizeof(A) == 4);

struct B { };
SA(1, sizeof(B) == 1);

struct C : A, B { };
SA(2, sizeof(C) == 4);

struct D { };
struct E : D { };
struct F : E { };

struct G : E, F { };
SA(3, sizeof(G) == 2);
