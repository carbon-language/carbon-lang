// RUN: clang-cc %s -triple i386-pc-linux-gnu -target-feature=+sse2 -verify -fsyntax-only

// PR3678
int test8() {
  asm("%0" : : "Yt"(1.0));
  asm("%0" : : "Yy"(1.0)); // expected-error {{invalid input constraint 'Yy' in asm}}
}
