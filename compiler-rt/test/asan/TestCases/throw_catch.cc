// RUN: %clangxx_asan -O %s -o %t && %run %t

// Clang doesn't support exceptions on Windows yet.
// XFAIL: win32

#include <assert.h>
#include <setjmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sanitizer/asan_interface.h>

__attribute__((noinline))
void Throw() {
  int local;
  fprintf(stderr, "Throw:  %p\n", &local);
  throw 1;
}

__attribute__((noinline))
void ThrowAndCatch() {
  int local;
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch:  %p\n", &local);
  }
}

void TestThrow() {
  char x[32];
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  ThrowAndCatch();
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  // FIXME: Invert this assertion once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  assert(!__asan_address_is_poisoned(x + 32));
}

void TestThrowInline() {
  char x[32];
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch\n");
  }
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  // FIXME: Invert this assertion once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  assert(!__asan_address_is_poisoned(x + 32));
}

static jmp_buf buf;

void TestLongJmp() {
  char x[32];
  fprintf(stderr, "\nTestLongJmp\n");
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  if (0 == setjmp(buf))
    longjmp(buf, 1);
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  // FIXME: Invert this assertion once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  assert(!__asan_address_is_poisoned(x + 32));
}

int main(int argc, char **argv) {
  TestThrow();
  TestThrowInline();
  TestLongJmp();
}
