// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,experimental.core.CastSize,unix.Malloc -analyzer-store=region -verify %s
#include "system-header-simulator.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *valloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *reallocf(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);

void myfoo(int *p);
void myfooint(int p);
char *fooRetPtr();

void f1() {
  int *p = malloc(12);
  return; // expected-warning{{Memory is never released; potential leak}}
}

void f2() {
  int *p = malloc(12);
  free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void f2_realloc_0() {
  int *p = malloc(12);
  realloc(p,0);
  realloc(p,0); // expected-warning{{Attempt to free released memory}}
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
    char x = *q; // expected-warning {{Memory is never released; potential leak}}
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
  free(p); // expected-warning {{Attempt to free released memory}}
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
  char *r = realloc(0, 12); // expected-warning {{Memory is never released; potential leak}}
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

void reallocRadar6337483_1() {
    char *buf = malloc(100);
    buf = (char*)realloc(buf, 0x1000000);
    if (!buf) {
        return;// expected-warning {{Memory is never released; potential leak}}
    }
    free(buf);
}

void reallocRadar6337483_2() {
    char *buf = malloc(100);
    char *buf2 = (char*)realloc(buf, 0x1000000);
    if (!buf2) { // expected-warning {{Memory is never released; potential leak}}
      ;
    } else {
      free(buf2);
    }
}

void reallocRadar6337483_3() {
    char * buf = malloc(100);
    char * tmp;
    tmp = (char*)realloc(buf, 0x1000000);
    if (!tmp) {
        free(buf);
        return;
    }
    buf = tmp;
    free(buf);
}

void reallocRadar6337483_4() {
    char *buf = malloc(100);
    char *buf2 = (char*)realloc(buf, 0x1000000);
    if (!buf2) {
      return;  // expected-warning {{Memory is never released; potential leak}}
    } else {
      free(buf2);
    }
}

int *reallocfTest1() {
  int *q = malloc(12);
  q = reallocf(q, 20);
  return q; // no warning - returning the allocated value
}

void reallocfRadar6337483_4() {
    char *buf = malloc(100);
    char *buf2 = (char*)reallocf(buf, 0x1000000);
    if (!buf2) {
      return;  // no warning - reallocf frees even on failure
    } else {
      free(buf2);
    }
}

void reallocfRadar6337483_3() {
    char * buf = malloc(100);
    char * tmp;
    tmp = (char*)reallocf(buf, 0x1000000);
    if (!tmp) {
        free(buf); // expected-warning {{Attempt to free released memory}}
        return;
    }
    buf = tmp;
    free(buf);
}

void reallocfPtrZero1() {
  char *r = reallocf(0, 12); // expected-warning {{Memory is never released; potential leak}}
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
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void f7_realloc() {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
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
  free(p); // expected-warning{{Attempt to free released memory}}
}

void mallocEscapeFreeUse() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myfoo(p); // expected-warning{{Use of memory after it is freed}}
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
  myfoo(x); // expected-warning{{Use of memory after it is freed}}
}

void mallocEscapeMalloc() {
  int *p = malloc(12);
  myfoo(p);
  p = malloc(12); // expected-warning{{Memory is never released; potential leak}}
}

void mallocMalloc() {
  int *p = malloc(12);
  p = malloc(12); // expected-warning 2 {{Memory is never released; potential leak}}
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
  myfoo(p); //expected-warning{{Use of memory after it is freed}}
}

void mallocFreeUse_params2() {
  int *p = malloc(12);
  free(p);
  myfooint(*p); //expected-warning{{Use of memory after it is freed}}
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
  return p; // expected-warning {{Use of memory after it is freed}}
}

int useAfterFreeStruct() {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  px->g = 5;
  free(px);
  return px->g; // expected-warning {{Use of memory after it is freed}}
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
    return; // expected-warning {{Memory is never released; potential leak}}
}

void mallocAssignment() {
  char *p = malloc(12);
  p = fooRetPtr(); // expected-warning {{leak}}
}

int vallocTest() {
  char *mem = valloc(12);
  return 0; // expected-warning {{Memory is never released; potential leak}}
}

void vallocEscapeFreeUse() {
  int *p = valloc(12);
  myfoo(p);
  free(p);
  myfoo(p); // expected-warning{{Use of memory after it is freed}}
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

char *ArrayG[12];

void globalArrayTest() {
  char *p = (char*)malloc(12);
  ArrayG[0] = p;
}

// Make sure that we properly handle a pointer stored into a local struct/array.
typedef struct _StructWithPtr {
  int *memP;
} StructWithPtr;

static StructWithPtr arrOfStructs[10];

void testMalloc() {
  int *x = malloc(12);
  StructWithPtr St;
  St.memP = x;
  arrOfStructs[0] = St;
}

StructWithPtr testMalloc2() {
  int *x = malloc(12);
  StructWithPtr St;
  St.memP = x;
  return St;
}

int *testMalloc3() {
  int *x = malloc(12);
  int *y = x;
  return y;
}

void testElemRegion1() {
  char *x = (void*)malloc(2);
  int *ix = (int*)x;
  free(&(x[0]));
}

void testElemRegion2(int **pp) {
  int *p = malloc(12);
  *pp = p;
  free(pp[0]);
}

void testElemRegion3(int **pp) {
  int *p = malloc(12);
  *pp = p;
  free(*pp);
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
  return px; // expected-warning {{Memory is never released; potential leak}}
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
  return 0;// expected-warning {{Memory is never released; potential leak}}
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

// Rely on the CString checker evaluation of the strcpy API to convey that the result of strcpy is equal to p.
void symbolLostWithStrcpy(char *s) {
  char *p = malloc(12);
  p = strcpy(p, s);
  free(p);
}


// The same test as the one above, but with what is actually generated on a mac.
static __inline char *
__inline_strcpy_chk (char *restrict __dest, const char *restrict __src)
{
  return __builtin___strcpy_chk (__dest, __src, __builtin_object_size (__dest, 2 > 1));
}

void symbolLostWithStrcpy_InlineStrcpyVersion(char *s) {
  char *p = malloc(12);
  p = ((__builtin_object_size (p, 0) != (size_t) -1) ? __builtin___strcpy_chk (p, s, __builtin_object_size (p, 2 > 1)) : __inline_strcpy_chk (p, s));
  free(p);
}

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
  return p;
}

// Potentially, the user could free the struct by performing pointer arithmetic on the return value.
// This is a variation of the specialMalloc issue, though probably would be more rare in correct code.
int *specialMallocWithStruct() {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  return &(px->g);
}

// Test various allocation/deallocation functions.

char *strdup(const char *s);
char *strndup(const char *s, size_t n);

void testStrdup(const char *s, unsigned validIndex) {
  char *s2 = strdup(s);
  s2[validIndex + 1] = 'b';// expected-warning {{Memory is never released; potential leak}}
}

int testStrndup(const char *s, unsigned validIndex, unsigned size) {
  char *s2 = strndup(s, size);
  s2 [validIndex + 1] = 'b';
  if (s2[validIndex] != 'a')
    return 0;
  else
    return 1;// expected-warning {{Memory is never released; potential leak}}
}

void testStrdupContentIsDefined(const char *s, unsigned validIndex) {
  char *s2 = strdup(s);
  char result = s2[1];// no warning
  free(s2);
}

// ----------------------------------------------------------------------------
// Test the system library functions to which the pointer can escape.
// This tests false positive suppression.

// For now, we assume memory passed to pthread_specific escapes.
// TODO: We could check that if a new pthread binding is set, the existing
// binding must be freed; otherwise, a memory leak can occur.
void testPthereadSpecificEscape(pthread_key_t key) {
  void *buf = malloc(12);
  pthread_setspecific(key, buf); // no warning
}

// PR12101: Test funopen().
static int releasePtr(void *_ctx) {
    free(_ctx);
    return 0;
}
FILE *useFunOpen() {
    void *ctx = malloc(sizeof(int));
    FILE *f = funopen(ctx, 0, 0, 0, releasePtr); // no warning
    if (f == 0) {
        free(ctx);
    }
    return f;
}
FILE *useFunOpenNoReleaseFunction() {
    void *ctx = malloc(sizeof(int));
    FILE *f = funopen(ctx, 0, 0, 0, 0);
    if (f == 0) {
        free(ctx);
    }
    return f; // expected-warning{{leak}}
}

// Test setbuf, setvbuf.
int my_main_no_warning() {
    char *p = malloc(100);
    setvbuf(stdout, p, 0, 100);
    return 0;
}
int my_main_no_warning2() {
    char *p = malloc(100);
    setbuf(__stdoutp, p);
    return 0;
}
int my_main_warn(FILE *f) {
    char *p = malloc(100);
    setvbuf(f, p, 0, 100);
    return 0;// expected-warning {{leak}}
}

// <rdar://problem/10978247>.
// some people use stack allocated memory as an optimization to avoid
// a heap allocation for small work sizes.  This tests the analyzer's
// understanding that the malloc'ed memory is not the same as stackBuffer.
void radar10978247(int myValueSize) {
  char stackBuffer[128];
  char *buffer;

  if (myValueSize <= sizeof(stackBuffer))
    buffer = stackBuffer;
  else 
    buffer = malloc(myValueSize);

  // do stuff with the buffer
  if (buffer != stackBuffer)
    free(buffer);
}

void radar10978247_positive(int myValueSize) {
  char stackBuffer[128];
  char *buffer;

  if (myValueSize <= sizeof(stackBuffer))
    buffer = stackBuffer;
  else 
    buffer = malloc(myValueSize);

  // do stuff with the buffer
  if (buffer == stackBuffer) // expected-warning {{leak}}
    return;
}

// <rdar://problem/11269741> Previously this triggered a false positive
// because malloc() is known to return uninitialized memory and the binding
// of 'o' to 'p->n' was not getting propertly handled.  Now we report a leak.
struct rdar11269741_a_t {
  struct rdar11269741_b_t {
    int m;
  } n;
};

int rdar11269741(struct rdar11269741_b_t o)
{
  struct rdar11269741_a_t *p = (struct rdar11269741_a_t *) malloc(sizeof(*p));
  p->n = o;
  return p->n.m; // expected-warning {{leak}}
}

// ----------------------------------------------------------------------------
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
    return; // expected-warning{{Memory is never released; potential leak}}
  return;
}

// ----------------------------------------------------------------------------
// False negatives.

// TODO: This requires tracking symbols stored inside the structs/arrays.
void testMalloc5() {
  StructWithPtr St;
  StructWithPtr *pSt = &St;
  pSt->memP = malloc(12);
}

// TODO: This is another false negative.
void testMallocWithParam(int **p) {
  *p = (int*) malloc(sizeof(int));
  *p = 0;
}

void testMallocWithParam_2(int **p) {
  *p = (int*) malloc(sizeof(int));
}

// TODO: This should produce a warning, similar to the previous issue.
void localArrayTest() {
  char *p = (char*)malloc(12);
  char *ArrayL[12];
  ArrayL[0] = p;
}

