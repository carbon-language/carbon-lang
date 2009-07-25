// RUN: clang-cc -fsyntax-only -verify %s 

static int main() { // expected-error {{'main' is not allowed to be declared static}}
}
