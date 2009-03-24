// RUN: clang-cc %s -fsyntax-only -verify -pedantic

typedef const char rchar;
int a(char* a, rchar* b) {
  return a-b;
}

// <rdar://problem/6520707> 
void f0(void (*fp)(void)) {
  int x = fp - fp; // expected-warning{{arithmetic on pointer to function type 'void (*)(void)' is a GNU extension}}
}
