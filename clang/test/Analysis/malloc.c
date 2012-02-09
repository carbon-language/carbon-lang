// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,experimental.core.CastSize,experimental.unix.Malloc -analyzer-store=region -verify %s
typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);

void myfoo(int *p);
void myfooint(int p);

void f1() {
  int *p = malloc(12);
  return; // expected-warning{{Allocated memory never released. Potential memory leak.}}
}

void f2() {
  int *p = malloc(12);
  free(p);
  free(p); // expected-warning{{Try to free a memory block that has been released}}
}

void f2_realloc_0() {
  int *p = malloc(12);
  realloc(p,0);
  realloc(p,0); // expected-warning{{Try to free a memory block that has been released}}
}

void f2_realloc_1() {
  int *p = malloc(12);
  int *q = realloc(p,0); // no-warning
}

// This case tests that storing malloc'ed memory to a static variable which is
// then returned is not leaked.  In the absence of known contracts for functions
// or inter-procedural analysis, this is a conservative answer.
int *f3() {
  static int *p = 0;
  p = malloc(12); 
  return p; // no-warning
}

// This case tests that storing malloc'ed memory to a static global variable
// which is then returned is not leaked.  In the absence of known contracts for
// functions or inter-procedural analysis, this is a conservative answer.
static int *p_f4 = 0;
int *f4() {
  p_f4 = malloc(12); 
  return p_f4; // no-warning
}

int *f5() {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no-warning
}

void f6() {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    free(p);
}

void f6_realloc() {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    realloc(p,0);
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

void f7_realloc() {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use dynamically allocated memory after it is freed.}}
}

void PR6123() {
  int *x = malloc(11); // expected-warning{{Cast a region whose size is not a multiple of the destination type size.}}
}

void PR7217() {
  int *buf = malloc(2); // expected-warning{{Cast a region whose size is not a multiple of the destination type size.}}
  buf[1] = 'c'; // not crash
}

void mallocCastToVoid() {
  void *p = malloc(2);
  const void *cp = p; // not crash
  free(p);
}

void mallocCastToFP() {
  void *p = malloc(2);
  void (*fp)() = p; // not crash
  free(p);
}

// This tests that malloc() buffers are undefined by default
char mallocGarbage () {
	char *buf = malloc(2);
	char result = buf[1]; // expected-warning{{undefined}}
	free(buf);
	return result;
}

// This tests that calloc() buffers need to be freed
void callocNoFree () {
  char *buf = calloc(2,2);
  return; // expected-warning{{never released}}
}

// These test that calloc() buffers are zeroed by default
char callocZeroesGood () {
	char *buf = calloc(2,2);
	char result = buf[3]; // no-warning
	if (buf[1] == 0) {
	  free(buf);
	}
	return result; // no-warning
}

char callocZeroesBad () {
	char *buf = calloc(2,2);
	char result = buf[3]; // no-warning
	if (buf[1] != 0) {
	  free(buf); // expected-warning{{never executed}}
	}
	return result; // expected-warning{{never released}}
}

void nullFree() {
  int *p = 0;
  free(p); // no warning - a nop
}

void paramFree(int *p) {
  myfoo(p);
  free(p); // no warning
  myfoo(p); // TODO: This should be a warning.
}

int* mallocEscapeRet() {
  int *p = malloc(12);
  return p; // no warning
}

void mallocEscapeFoo() {
  int *p = malloc(12);
  myfoo(p);
  return; // no warning
}

void mallocEscapeFree() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
}

void mallocEscapeFreeFree() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  free(p); // expected-warning{{Try to free a memory block that has been released}}
}

void mallocEscapeFreeUse() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myfoo(p); // expected-warning{{Use dynamically allocated memory after it is freed.}}
}

int *myalloc();
void myalloc2(int **p);

void mallocEscapeFreeCustomAlloc() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  p = myalloc();
  free(p); // no warning
}

void mallocEscapeFreeCustomAlloc2() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myalloc2(&p);
  free(p); // no warning
}

void mallocBindFreeUse() {
  int *x = malloc(12);
  int *y = x;
  free(y);
  myfoo(x); // expected-warning{{Use dynamically allocated memory after it is freed.}}
}

void mallocEscapeMalloc() {
  int *p = malloc(12);
  myfoo(p);
  p = malloc(12); // expected-warning{{Allocated memory never released. Potential memory leak.}}
}

void mallocMalloc() {
  int *p = malloc(12);
  p = malloc(12); // expected-warning{{Allocated memory never released. Potential memory leak}}
}

void mallocFreeMalloc() {
  int *p = malloc(12);
  free(p);
  p = malloc(12);
  free(p);
}

void mallocFreeUse_params() {
  int *p = malloc(12);
  free(p);
  myfoo(p); //expected-warning{{Use dynamically allocated memory after it is freed}}
  myfooint(*p); //expected-warning{{Use dynamically allocated memory after it is freed}}
}

void mallocFailedOrNot() {
  int *p = malloc(12);
  if (!p)
    free(p);
  else
    free(p);
}

int *Gl;
struct GlStTy {
  int *x;
};

struct GlStTy GlS = {0};

void GlobalFree() {
  free(Gl);
}

void GlobalMalloc() {
  Gl = malloc(12);
}

void GlobalStructMalloc() {
  int *a = malloc(12);
  GlS.x = a;
}

void GlobalStructMallocFree() {
  int *a = malloc(12);
  GlS.x = a;
  free(GlS.x);
}
