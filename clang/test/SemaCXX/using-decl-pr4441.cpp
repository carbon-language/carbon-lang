// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
    struct B { };
    void operator+(B,B);
}

using A::operator+;
