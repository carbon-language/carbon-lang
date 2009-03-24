// RUN: clang-cc -fsyntax-only -pedantic -verify %s

int test(char *C) { // nothing here should warn.
  return C != ((void*)0);
  return C != (void*)0;
  return C != 0;
}

int equal(char *a, const char *b)
{
    return a == b;
}

int arrays(char (*a)[5], char(*b)[10], char(*c)[5]) {
  int d = (a == c);
  return a == b; // expected-warning {{comparison of distinct pointer types}}
}
