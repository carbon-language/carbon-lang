// RUN: clang-cc -fsyntax-only -verify %s

namespace A {
    struct B { };
    void operator+(B,B);
}

using A::operator+;
