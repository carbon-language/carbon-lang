// RUN: clang -analyze -checker-simple -verify %s
//
// Just exercise the analyzer (no assertions).


static const char * f1(const char *x, char *y) {
  while (*x != 0) {
    *y++ = *x++;
  }
}
