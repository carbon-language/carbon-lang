// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DCHECK -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s
// RUN: %clang_cc1 -analyze -DUSE_BUILTINS -DCHECK -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-experimental-checks -verify %s

//===----------------------------------------------------------------------===
// Declarations
//===----------------------------------------------------------------------===

// Some functions having a checking variant, which checks if there is overflow
// using a flow-insensitive calculation of the buffer size. If CHECK is defined,
// use those instead to make sure they are still checked by the analyzer.

// Some functions are implemented as builtins. These should be #defined as
// BUILTIN(f), which will prepend "__builtin_" if USE_BUILTINS is defined.

// Functions that have both checking and builtin variants should be declared
// carefully! See memcpy() for an example.

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else /* USE_BUILTINS */
# define BUILTIN(f) f
#endif /* USE_BUILTINS */

typedef typeof(sizeof(int)) size_t;

//===----------------------------------------------------------------------===
// memcpy()
//===----------------------------------------------------------------------===

#ifdef CHECK

#define __memcpy_chk BUILTIN(__memcpy_chk)
void *__memcpy_chk(void *restrict s1, const void *restrict s2, size_t n,
                   size_t destlen);

#define memcpy(a,b,c) __memcpy_chk(a,b,c,(size_t)-1)

#else /* CHECK */

#define memcpy BUILTIN(memcpy)
void *memcpy(void *restrict s1, const void *restrict s2, size_t n);

#endif /* CHECK */


void memcpy0 () {
  char src[] = {1, 2, 3, 4};
  char dst[4];

  memcpy(dst, src, 4); // no-warning

  if (memcpy(dst, src, 4) != dst) {
    (void)*(char*)0; // no-warning -- should be unreachable
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

//===----------------------------------------------------------------------===
// memmove()
//===----------------------------------------------------------------------===

#ifdef CHECK

#define __memmove_chk BUILTIN(__memmove_chk)
void *__memmove_chk(void *s1, const void *s2, size_t n, size_t destlen);

#define memmove(a,b,c) __memmove_chk(a,b,c,(size_t)-1)

#else /* CHECK */

#define memmove BUILTIN(memmove)
void *memmove(void *s1, const void *s2, size_t n);

#endif /* CHECK */


void memmove0 () {
  char src[] = {1, 2, 3, 4};
  char dst[4];

  memmove(dst, src, 4); // no-warning

  if (memmove(dst, src, 4) != dst) {
    (void)*(char*)0; // no-warning -- should be unreachable
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
