// RUN: clang-cc -fsyntax-only -verify %s
enum E {
  Val1,
  Val2
};

int& enumerator_type(int);
float& enumerator_type(E);

void f() {
  E e = Val1;
  float& fr = enumerator_type(Val2);
}

// <rdar://problem/6502934>
typedef enum Foo {
  A = 0,
  B = 1
} Foo;

void bar() {
  Foo myvar = A;
  myvar = B;
}

/// PR3688
struct s1 {
  enum e1 (*bar)(void); // expected-error{{ISO C++ forbids forward references to 'enum' types}} expected-note{{forward declaration of 'enum s1::e1'}}
};

enum e1 { YES, NO };

static enum e1 badfunc(struct s1 *q) {
  return q->bar(); // expected-error{{calling function with incomplete return type 'enum s1::e1'}}
}

enum e2; // expected-error{{ISO C++ forbids forward references to 'enum' types}}
