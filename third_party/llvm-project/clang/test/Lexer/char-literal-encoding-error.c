// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -x c++ %s

// This file is encoded using ISO-8859-1

int main() {
  (void)'�'; // expected-warning {{illegal character encoding in character literal}}
  (void)u'�'; // expected-error {{illegal character encoding in character literal}}
  (void)U'�'; // expected-error {{illegal character encoding in character literal}}
  (void)L'�'; // expected-error {{illegal character encoding in character literal}}

  // For narrow character literals, since there is no error, make sure the
  // encoding is correct
  static_assert((unsigned char)'�' == 0xE9, ""); // expected-warning {{illegal character encoding in character literal}}
  static_assert('��' == 0xE9E9, ""); // expected-warning {{illegal character encoding in character literal}} expected-warning {{multi-character character constant}}
}
