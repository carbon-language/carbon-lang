// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -x c++ %s

// This file is encoded using ISO-8859-1

int main() {
  'é'; // expected-error {{illegal character encoding in character literal}}
  u'é'; // expected-error {{illegal character encoding in character literal}}
  U'é'; // expected-error {{illegal character encoding in character literal}}
  L'é'; // expected-error {{illegal character encoding in character literal}}
}
