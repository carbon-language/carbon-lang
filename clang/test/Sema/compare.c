// RUN: clang -parse-ast-check %s

int test(char *C) { // nothing here should warn.
  return C != ((void*)0);
  return C != (void*)0;
  return C != 0;
}

