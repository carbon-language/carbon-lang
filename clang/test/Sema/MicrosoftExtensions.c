// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions


struct A
{
   int a[];  /* expected-warning {{flexible array member 'a' in otherwise empty struct is a Microsoft extension}} */
};

struct C {
   int l;
   union {
       int c1[];   /* expected-warning {{flexible array member 'c1' in a union is a Microsoft extension}}  */
       char c2[];  /* expected-warning {{flexible array member 'c2' in a union is a Microsoft extension}} */
   };
};


struct D {
   int l;
   int D[];
};
