// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.unix,core.uninitialized -analyzer-store=region -verify %s
typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

char stackBased1 () {
  char buf[2];
  buf[0] = 'a';
  return buf[1]; // expected-warning{{Undefined}}
}

char stackBased2 () {
  char buf[2];
  buf[1] = 'a';
  return buf[0]; // expected-warning{{Undefined}}
}

char heapBased1 () {
  char *buf = malloc(2);
  buf[0] = 'a';
  char result = buf[1]; // expected-warning{{undefined}}
  free(buf);
  return result;
}

char heapBased2 () {
  char *buf = malloc(2);
  buf[1] = 'a';
  char result = buf[0]; // expected-warning{{undefined}}
  free(buf);
  return result;
}
