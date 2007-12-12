// RUN: clang %s -fsyntax-only -verify

void test() {
    char = 4;  // expected-error {{expected identifier}} expected-error{{declarator requires an identifier}}

}


