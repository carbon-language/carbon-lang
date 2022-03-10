// RUN: %clang_cc1 -std=c++11 -verify %s

extern "C" {
  extern R"(C++)" { }
}

#define plusplus "++"
extern "C" plusplus {
}

extern u8"C" {} // expected-error {{string literal in language linkage specifier cannot have an encoding-prefix}}
extern L"C" {} // expected-error {{string literal in language linkage specifier cannot have an encoding-prefix}}
extern u"C++" {} // expected-error {{string literal in language linkage specifier cannot have an encoding-prefix}}
extern U"C" {} // expected-error {{string literal in language linkage specifier cannot have an encoding-prefix}}
