// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=range -verify %s

void f1() {
  int const &i = 3;
  int b = i;

  int *p = 0;

  if (b != 3)
    *p = 1; // no-warning
}

char* ptr();
char& ref();

// These next two tests just shouldn't crash.
char t1 () {
  ref() = 'c';
  return '0';
}

// just a sanity test, the same behavior as t1()
char t2 () {
  *ptr() = 'c';
  return '0';
}
