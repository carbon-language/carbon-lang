// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core -analyzer-store=region -verify %s

void fill_r (int * const &x);

char testPointer () {
  int x[8];
  int *xp = x;
  fill_r(xp);

  return x[0]; // no-warning
}

char testArray () {
  int x[8];
  fill_r(x);

  return x[0]; // no-warning
}

char testReferenceCast () {
  int x[8];
  int *xp = x;
  fill_r(reinterpret_cast<int * const &>(xp));
  
  return x[0]; // no-warning
}


void fill (int *x);
char testReferenceCastRValue () {
  int x[8];
  int *xp = x;
  fill(reinterpret_cast<int * const &>(xp));

  return x[0]; // no-warning
}
