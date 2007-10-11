// RUN: clang -fsyntax-only -verify %s

int test(char *C) { // nothing here should warn.
  return C != ((void*)0);
  return C != (void*)0;
  return C != 0;
}

int equal(char *a, const char *b)
{
    return a == b;
}
