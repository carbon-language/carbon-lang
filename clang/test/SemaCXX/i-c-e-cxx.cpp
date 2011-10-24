// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

// C++-specific tests for integral constant expressions.

const int c = 10;
int ar[c];

struct X0 {
  static const int value = static_cast<int>(4.0);
};

void f() {
  if (const int value = 17) {
    int array[value];
  }
}

int a() {
  const int t=t;
  switch(1) { // expected-warning {{no case matching constant switch condition '1'}}
    case t:; // expected-error {{not an integer constant expression}}
  }
}

// PR6206:  out-of-line definitions are legit
namespace pr6206 {
  class Foo {
  public:
    static const int kBar;
  };

  const int Foo::kBar = 20;
  
  char Test() {
    char str[Foo::kBar];
    str[0] = '0';
    return str[0];
  }
}

// PR6373:  default arguments don't count.
void pr6373(const unsigned x = 0) {
  unsigned max = 80 / x;
}


// rdar://9204520
namespace rdar9204520 {
  
struct A {
  static const int B = int(0.75 * 1000 * 1000); // expected-warning {{not a constant expression, accepted as an extension}}
};

int foo() { return A::B; }
}

// PR11040
const int x = 10;
int* y = reinterpret_cast<const char&>(x); // expected-error {{cannot initialize}}

// This isn't an integral constant expression, but make sure it folds anyway.
struct PR8836 { char _; long long a; }; // expected-warning {{long long}}
int PR8836test[(__typeof(sizeof(int)))&reinterpret_cast<const volatile char&>((((PR8836*)0)->a))]; // expected-warning {{folded to constant array as an extension}}

const int nonconst = 1.0;
int arr[nonconst]; // expected-warning {{folded to constant array as an extension}}
const int castfloat = static_cast<int>(1.0);
int arr2[castfloat]; // ok

extern const int &Recurse1;
const int &Recurse2 = Recurse1, &Recurse1 = Recurse2;
