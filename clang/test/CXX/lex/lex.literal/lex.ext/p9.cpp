// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

using size_t = decltype(sizeof(int));
void operator "" _x(const wchar_t *, size_t);

namespace std_example {

int main() {
  // FIXME: once we implement the semantics of literal operators, this warning
  // should vanish.
  L"A" "B" "C"_x; // expected-warning {{expression result unused}}
  "P"_x "Q" "R"_y; // expected-error {{differing user-defined suffixes ('_x' and '_y') in string literal concatenation}}
}

}
