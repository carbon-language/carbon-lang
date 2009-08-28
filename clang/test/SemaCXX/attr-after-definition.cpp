// RUN: clang-cc -fsyntax-only -verify %s
struct X { };
struct Y { };

bool f0(X) { return true; } // expected-note{{definition}}
bool f1(X) { return true; }

__attribute__ ((__visibility__("hidden"))) bool f0(X); // expected-warning{{attribute}}
__attribute__ ((__visibility__("hidden"))) bool f1(Y);
