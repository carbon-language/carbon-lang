// RUN: clang -fsyntax-only -verify %s

void abcdefghi12(void) {
 const char (*ss)[12] = &__func__;
 static int arr[sizeof(__func__)==12 ? 1 : -1];
}

char *X = __func__; // expected-error {{predefined identifier is only valid}}

void a() {
  __func__[0] = 'a';  // expected-error {{variable is not assignable}}
}
