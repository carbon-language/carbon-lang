// RUN: clang -fsyntax-only -verify %s -std=c++0x

int f();

static_assert(f(), "f"); // expected-error {{static_assert expression is not an integral constant expression}}
static_assert(true, "true is not false");
static_assert(false, "false is false"); // expected-error {{static_assert failed "false is false"}}

void g() {
    static_assert(false, "false is false"); // expected-error {{static_assert failed "false is false"}}
}

class C {
    static_assert(false, "false is false"); // expected-error {{static_assert failed "false is false"}}
};

template<int N> struct T {
    static_assert(N == 2, "N is not 2!");
};

template<typename T> struct S {
    static_assert(sizeof(T) > sizeof(char), "Type not big enough!");
};

