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

struct Empty { Empty(); };

struct I : Empty { 
  Empty e;
};
SA(4, sizeof(I) == 2);

struct J : Empty { 
  Empty e[2];
};
SA(5, sizeof(J) == 3);

template<int N> struct Derived : Empty, Derived<N - 1> { 
};
template<> struct Derived<0> : Empty { };

struct S1 : virtual Derived<10> { 
  Empty e;
};
SA(6, sizeof(S1) == 24);

struct S2 : virtual Derived<10> { 
  Empty e[2];
};
SA(7, sizeof(S2) == 24);
