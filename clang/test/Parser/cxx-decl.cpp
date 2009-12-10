// RUN: clang-cc -verify -fsyntax-only %s

int x(*g); // expected-error {{use of undeclared identifier 'g'}}

struct Type { };


// PR4451 - We should recover well from the typo of '::' as ':' in a2.
namespace y {
  struct a { };  
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

