// RUN: clang-cc -fsyntax-only -verify %s 

// Bool literals can be enum values.
enum {
  ReadWrite = false,
  ReadOnly = true
};

// bool cannot be decremented, and gives a warning on increment
void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
  b++; // expected-warning {{incrementing expression of type bool is deprecated}}
  --b; // expected-error {{cannot decrement expression of type bool}}
  b--; // expected-error {{cannot decrement expression of type bool}}
}
