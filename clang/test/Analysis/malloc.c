// RUN: clang -cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-experimental-checks -analyzer-store=region -verify %s
typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);

void f1() {
  int *p = malloc(10);
  return; // expected-warning{{Allocated memory never released. Potential memory leak.}}
}

void f1_b() {
  int *p = malloc(10); // expected-warning{{Allocated memory never released. Potential memory leak.}}
}

void f2() {
  int *p = malloc(10);
  free(p);
  free(p); // expected-warning{{Try to free a memory block that has been released}}
}

// This case tests that storing malloc'ed memory to a static variable which is
// then returned is not leaked.  In the absence of known contracts for functions
// or inter-procedural analysis, this is a conservative answer.
int *f3() {
  static int *p = 0;
  p = malloc(10); 
  return p; // no-warning
}

// This case tests that storing malloc'ed memory to a static global variable
// which is then returned is not leaked.  In the absence of known contracts for
// functions or inter-procedural analysis, this is a conservative answer.
static int *p_f4 = 0;
int *f4() {
  p_f4 = malloc(10); 
  return p_f4; // no-warning
}

int *f5() {
  int *q = malloc(10);
  q = realloc(q, 20);
  return q; // no-warning
}
