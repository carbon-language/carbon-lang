// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=range -verify %s
// XFAIL: *
typedef typeof(sizeof(int)) size_t;
void malloc (size_t);

void f1() {
  int const &i = 3;  // <--- **FIXME** This is currently not being modeled correctly.
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

// Each of the tests below is repeated with pointers as well as references.
// This is mostly a sanity check, but then again, both should work!
char t3 () {
  char& r = ref();
  r = 'c'; // no-warning
  if (r) return r;
  return *(char*)0; // no-warning
}

char t4 () {
  char* p = ptr();
  *p = 'c'; // no-warning
  if (*p) return *p;
  return *(char*)0; // no-warning
}

char t5 (char& r) {
  r = 'c'; // no-warning
  if (r) return r;
  return *(char*)0; // no-warning
}

char t6 (char* p) {
  *p = 'c'; // no-warning
  if (*p) return *p;
  return *(char*)0; // no-warning
}
