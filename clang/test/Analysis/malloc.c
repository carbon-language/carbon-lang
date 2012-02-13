// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,experimental.core.CastSize,experimental.unix.Malloc -analyzer-store=region -verify %s
#include "system-header-simulator.h"

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

void reallocNotNullPtr(unsigned sizeIn) {
  unsigned size = 12;
  char *p = (char*)malloc(size);
  if (p) {
    char *q = (char*)realloc(p, sizeIn);
    char x = *q; // expected-warning {{Allocated memory never released.}}
  }
}

int *realloctest1() {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no warning - returning the allocated value
}

// p should be freed if realloc fails.
void reallocFails() {
  char *p = malloc(12);
  char *r = realloc(p, 12+1);
  if (!r) {
    free(p);
  } else {
    free(r);
  }
}

void reallocSizeZero1() {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  if (!r) {
    free(p);
  } else {
    free(r);
  }
}

void reallocSizeZero2() {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  if (!r) {
    free(p);
  } else {
    free(r);
  }
  free(p); // expected-warning {{Try to free a memory block that has been released}}
}

void reallocSizeZero3() {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  free(r);
}

void reallocSizeZero4() {
  char *r = realloc(0, 0);
  free(r);
}

void reallocSizeZero5() {
  char *r = realloc(0, 0);
}

void reallocPtrZero1() {
  char *r = realloc(0, 12); // expected-warning {{Allocated memory never released.}}
}

void reallocPtrZero2() {
  char *r = realloc(0, 12);
  if (r)
    free(r);
}

void reallocPtrZero3() {
  char *r = realloc(0, 12);
  free(r);
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
  x[0] = 'a'; // expected-warning{{Use of dynamically allocated memory after it is freed.}}
}

void f7_realloc() {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use of dynamically allocated memory after it is freed.}}
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
  myfoo(p); // expected-warning{{Use of dynamically allocated memory after it is freed.}}
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
  myfoo(x); // expected-warning{{Use of dynamically allocated memory after it is freed.}}
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
  myfoo(p); //expected-warning{{Use of dynamically allocated memory after it is freed}}
}

void mallocFreeUse_params2() {
  int *p = malloc(12);
  free(p);
  myfooint(*p); //expected-warning{{Use of dynamically allocated memory after it is freed}}
}

void mallocFailedOrNot() {
  int *p = malloc(12);
  if (!p)
    free(p);
  else
    free(p);
}

struct StructWithInt {
  int g;
};

int *mallocReturnFreed() {
  int *p = malloc(12);
  free(p);
  return p; // expected-warning {{Use of dynamically allocated}}
}

int useAfterFreeStruct() {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  px->g = 5;
  free(px);
  return px->g; // expected-warning {{Use of dynamically allocated}}
}

void nonSymbolAsFirstArg(int *pp, struct StructWithInt *p);

void mallocEscapeFooNonSymbolArg() {
  struct StructWithInt *p = malloc(sizeof(struct StructWithInt));
  nonSymbolAsFirstArg(&p->g, p);
  return; // no warning
}

void mallocFailedOrNotLeak() {
  int *p = malloc(12);
  if (p == 0)
    return; // no warning
  else
    return; // expected-warning {{Allocated memory never released. Potential memory leak.}}
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

// Region escape testing.

unsigned takePtrToPtr(int **p);
void PassTheAddrOfAllocatedData(int f) {
  int *p = malloc(12);
  // We don't know what happens after the call. Should stop tracking here.
  if (takePtrToPtr(&p))
    f++;
  free(p); // no warning
}

struct X {
  int *p;
};
unsigned takePtrToStruct(struct X *s);
int ** foo2(int *g, int f) {
  int *p = malloc(12);
  struct X *px= malloc(sizeof(struct X));
  px->p = p;
  // We don't know what happens after this call. Should not track px nor p.
  if (takePtrToStruct(px))
    f++;
  free(p);
  return 0;
}

struct X* RegInvalidationDetect1(struct X *s2) {
  struct X *px= malloc(sizeof(struct X));
  px->p = 0;
  px = s2;
  return px; // expected-warning {{Allocated memory never released. Potential memory leak.}}
}

struct X* RegInvalidationGiveUp1() {
  int *p = malloc(12);
  struct X *px= malloc(sizeof(struct X));
  px->p = p;
  return px;
}

int **RegInvalidationDetect2(int **pp) {
  int *p = malloc(12);
  pp = &p;
  pp++;
  return 0;// expected-warning {{Allocated memory never released. Potential memory leak.}}
}

extern void exit(int) __attribute__ ((__noreturn__));
void mallocExit(int *g) {
  struct xx *p = malloc(12);
  if (g != 0)
    exit(1);
  free(p);
  return;
}

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));
#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))
void mallocAssert(int *g) {
  struct xx *p = malloc(12);

  assert(g != 0);
  free(p);
  return;
}

void doNotInvalidateWhenPassedToSystemCalls(char *s) {
  char *p = malloc(12);
  strlen(p);
  strcpy(p, s); // expected-warning {{leak}}
}

// Below are the known false positives.

// TODO: There should be no warning here. This one might be difficult to get rid of.
void dependsOnValueOfPtr(int *g, unsigned f) {
  int *p;

  if (f) {
    p = g;
  } else {
    p = malloc(12);
  }

  if (p != g)
    free(p);
  else
    return; // expected-warning{{Allocated memory never released. Potential memory leak}}
  return;
}

// TODO: Should this be a warning?
// Here we are returning a pointer one past the allocated value. An idiom which
// can be used for implementing special malloc. The correct uses of this might
// be rare enough so that we could keep this as a warning.
static void *specialMalloc(int n){
  int *p;
  p = malloc( n+8 );
  if( p ){
    p[0] = n;
    p++;
  }
  return p;// expected-warning {{Allocated memory never released. Potential memory leak.}}
}
