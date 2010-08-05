// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DVARIANT -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -DVARIANT -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s

//===----------------------------------------------------------------------===
// Declarations
//===----------------------------------------------------------------------===

// Some functions are so similar to each other that they follow the same code
// path, such as memcpy and __memcpy_chk, or memcmp and bcmp. If VARIANT is
// defined, make sure to use the variants instead to make sure they are still
// checked by the analyzer.

// Some functions are implemented as builtins. These should be #defined as
// BUILTIN(f), which will prepend "__builtin_" if USE_BUILTINS is defined.

// Functions that have variants and are also availabe as builtins should be
// declared carefully! See memcpy() for an example.

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else /* USE_BUILTINS */
# define BUILTIN(f) f
#endif /* USE_BUILTINS */

typedef typeof(sizeof(int)) size_t;

//===----------------------------------------------------------------------===
// memcpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __memcpy_chk BUILTIN(__memcpy_chk)
void *__memcpy_chk(void *restrict s1, const void *restrict s2, size_t n,
                   size_t destlen);

#define memcpy(a,b,c) __memcpy_chk(a,b,c,(size_t)-1)

#else /* VARIANT */

#define memcpy BUILTIN(memcpy)
void *memcpy(void *restrict s1, const void *restrict s2, size_t n);

#endif /* VARIANT */


void memcpy0 () {
  char src[] = {1, 2, 3, 4};
  char dst[4];

  memcpy(dst, src, 4); // no-warning

  if (memcpy(dst, src, 4) != dst) {
    (void)*(char*)0; // no-warning
  }
}

void memcpy1 () {
  char src[] = {1, 2, 3, 4};
  char dst[10];

  memcpy(dst, src, 5); // expected-warning{{out-of-bound}}
}

void memcpy2 () {
  char src[] = {1, 2, 3, 4};
  char dst[1];

  memcpy(dst, src, 4); // expected-warning{{out-of-bound}}
}

void memcpy3 () {
  char src[] = {1, 2, 3, 4};
  char dst[3];

  memcpy(dst+1, src+2, 2); // no-warning
}

void memcpy4 () {
  char src[] = {1, 2, 3, 4};
  char dst[10];

  memcpy(dst+2, src+2, 3); // expected-warning{{out-of-bound}}
}

void memcpy5() {
  char src[] = {1, 2, 3, 4};
  char dst[3];

  memcpy(dst+2, src+2, 2); // expected-warning{{out-of-bound}}
}

void memcpy6() {
  int a[4] = {0};
  memcpy(a, a, 8); // expected-warning{{overlapping}}  
}

void memcpy7() {
  int a[4] = {0};
  memcpy(a+2, a+1, 8); // expected-warning{{overlapping}}
}

void memcpy8() {
  int a[4] = {0};
  memcpy(a+1, a+2, 8); // expected-warning{{overlapping}}
}

void memcpy9() {
  int a[4] = {0};
  memcpy(a+2, a+1, 4); // no-warning
  memcpy(a+1, a+2, 4); // no-warning
}

void memcpy10() {
  char a[4] = {0};
  memcpy(0, a, 4); // expected-warning{{Null pointer argument in call to byte string function}}
}

void memcpy11() {
  char a[4] = {0};
  memcpy(a, 0, 4); // expected-warning{{Null pointer argument in call to byte string function}}
}

void memcpy12() {
  char a[4] = {0};
  memcpy(0, a, 0); // no-warning
  memcpy(a, 0, 0); // no-warning
}

//===----------------------------------------------------------------------===
// memmove()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __memmove_chk BUILTIN(__memmove_chk)
void *__memmove_chk(void *s1, const void *s2, size_t n, size_t destlen);

#define memmove(a,b,c) __memmove_chk(a,b,c,(size_t)-1)

#else /* VARIANT */

#define memmove BUILTIN(memmove)
void *memmove(void *s1, const void *s2, size_t n);

#endif /* VARIANT */


void memmove0 () {
  char src[] = {1, 2, 3, 4};
  char dst[4];

  memmove(dst, src, 4); // no-warning

  if (memmove(dst, src, 4) != dst) {
    (void)*(char*)0; // no-warning
  }
}

void memmove1 () {
  char src[] = {1, 2, 3, 4};
  char dst[10];

  memmove(dst, src, 5); // expected-warning{{out-of-bound}}
}

void memmove2 () {
  char src[] = {1, 2, 3, 4};
  char dst[1];

  memmove(dst, src, 4); // expected-warning{{out-of-bound}}
}

//===----------------------------------------------------------------------===
// memcmp()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define bcmp BUILTIN(bcmp)
// __builtin_bcmp is not defined with const in Builtins.def.
int bcmp(/*const*/ void *s1, /*const*/ void *s2, size_t n);
#define memcmp bcmp

#else /* VARIANT */

#define memcmp BUILTIN(memcmp)
int memcmp(const void *s1, const void *s2, size_t n);

#endif /* VARIANT */


void memcmp0 () {
  char a[] = {1, 2, 3, 4};
  char b[4] = { 0 };

  memcmp(a, b, 4); // no-warning
}

void memcmp1 () {
  char a[] = {1, 2, 3, 4};
  char b[10] = { 0 };

  memcmp(a, b, 5); // expected-warning{{out-of-bound}}
}

void memcmp2 () {
  char a[] = {1, 2, 3, 4};
  char b[1] = { 0 };

  memcmp(a, b, 4); // expected-warning{{out-of-bound}}
}

void memcmp3 () {
  char a[] = {1, 2, 3, 4};

  if (memcmp(a, a, 4))
    (void)*(char*)0; // no-warning
}

void memcmp4 (char *input) {
  char a[] = {1, 2, 3, 4};

  if (memcmp(a, input, 4))
    (void)*(char*)0; // expected-warning{{null}}
}

void memcmp5 (char *input) {
  char a[] = {1, 2, 3, 4};

  if (memcmp(a, 0, 0)) // no-warning
    (void)*(char*)0;   // no-warning
  if (memcmp(0, a, 0)) // no-warning
    (void)*(char*)0;   // no-warning
  if (memcmp(a, input, 0)) // no-warning
    (void)*(char*)0;   // no-warning
}

void memcmp6 (char *a, char *b, size_t n) {
  int result = memcmp(a, b, n);
  if (result != 0)
    return;
  if (n == 0)
    (void)*(char*)0; // expected-warning{{null}}
}

int memcmp7 (char *a, size_t x, size_t y, size_t n) {
  // We used to crash when either of the arguments was unknown.
  return memcmp(a, &a[x*y], n) +
         memcmp(&a[x*y], a, n);
}

//===----------------------------------------------------------------------===
// bcopy()
//===----------------------------------------------------------------------===

#define bcopy BUILTIN(bcopy)
// __builtin_bcopy is not defined with const in Builtins.def.
void bcopy(/*const*/ void *s1, void *s2, size_t n);


void bcopy0 () {
  char src[] = {1, 2, 3, 4};
  char dst[4];

  bcopy(src, dst, 4); // no-warning
}

void bcopy1 () {
  char src[] = {1, 2, 3, 4};
  char dst[10];

  bcopy(src, dst, 5); // expected-warning{{out-of-bound}}
}

void bcopy2 () {
  char src[] = {1, 2, 3, 4};
  char dst[1];

  bcopy(src, dst, 4); // expected-warning{{out-of-bound}}
}
