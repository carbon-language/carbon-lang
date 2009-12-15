// RUN: %clang_cc1 -verify -fsyntax-only %s

int x(*g); // expected-error {{use of undeclared identifier 'g'}}

struct Type {
  int Type;
};


// PR4451 - We should recover well from the typo of '::' as ':' in a2.
namespace y {
  struct a { };
  typedef int b;
}

y::a a1;
y:a a2;  // expected-error {{unexpected ':' in nested name specifier}}
y::a a3 = a2;

// Some valid colons:
void foo() {
y:  // label
  y::a s;
  
  int a = 4;
  a = a ? a : a+1;
}

struct b : y::a {};

template <typename T>
class someclass {
  
  int bar() {
    T *P;
    return 1 ? P->x : P->y;
  }
};

enum { fooenum = 1 };

struct a {
  int Type : fooenum;
};

void test(struct Type *P) {
  int Type;
  Type = 1 ? P->Type : Type;
  
  Type = (y:b) 4;   // expected-error {{unexpected ':' in nested name specifier}}
  Type = 1 ? (
              (y:b)  // expected-error {{unexpected ':' in nested name specifier}}
              4) : 5;
}
