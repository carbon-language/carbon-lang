// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fixit -x c %t || true
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c %t
struct Point {
  float x, y;
};

struct Rectangle {
  struct Point top_left, // expected-note{{'top_left' declared here}}
               bottom_right;
};

enum Color { Red, Green, Blue };

struct Window {
  struct Rectangle bounds; // expected-note{{'bounds' declared here}}
  enum Color color;
};

struct Window window = {
  .bunds. // expected-error{{field designator 'bunds' does not refer to any field in type 'struct Window'; did you mean 'bounds'?}}
  topleft.x = 3.14, // expected-error{{field designator 'topleft' does not refer to any field in type 'struct Rectangle'; did you mean 'top_left'?}}
  2.71818, 5.0, 6.0, Red
};
