// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify 

#define SA(n, p) int a##n[(p) ? 1 : -1]

struct A {
  int a;
  char b;
};

SA(0, sizeof(A) == 8);

struct B : A {
  char c;
};

SA(1, sizeof(B) == 12);

struct C {
// Make fields private so C won't be a POD type.
private:
  int a;
  char b;
};

SA(2, sizeof(C) == 8);

struct D : C {
  char c;
};

SA(3, sizeof(D) == 8);

struct __attribute__((packed)) E {
  char b;
  int a;
};

SA(4, sizeof(E) == 5);

struct __attribute__((packed)) F : E {
  char d;
};

SA(5, sizeof(F) == 6);

struct G { G(); };
struct H : G { };

SA(6, sizeof(H) == 1);

// PR5580
namespace PR5580 {

class A { bool iv0 : 1; };
SA(7, sizeof(A) == 1);  

class B : A { bool iv0 : 1; };
SA(8, sizeof(B) == 2);

struct C { bool iv0 : 1; };
SA(9, sizeof(C) == 1);  

struct D : C { bool iv0 : 1; };
SA(10, sizeof(D) == 2);

}
