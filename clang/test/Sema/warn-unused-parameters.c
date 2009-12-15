// RUN: %clang -fblocks -fsyntax-only -Wunused-parameter %s -Xclang -verify

int f0(int x,
       int y, // expected-warning{{unused}}
       int z __attribute__((unused))) {
  return x;
}

void f1() {
  (void)^(int x,
          int y, // expected-warning{{unused}}
          int z __attribute__((unused))) { return x; };
}
