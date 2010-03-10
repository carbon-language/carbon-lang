// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-experimental-checks -analyzer-store=region -verify %s
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

void f6() {
  int *p = malloc(10);
  if (!p)
    return; // no-warning
  else
    free(p);
}

char *doit2();
void pr6069() {
  char *buf = doit2();
  free(buf);
}

void pr6293() {
  free(0);
}

void f7() {
  char *x = (char*) malloc(4);
  free(x);
  x[0] = 'a'; // expected-warning{{Use dynamically allocated memory after it is freed.}}
}
