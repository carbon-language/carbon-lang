// Clang doesn't support SEH on Windows yet, so for the time being we
// build this program in two parts: the code with SEH is built with CL,
// the rest is built with Clang.  This represents the typical scenario when we
// build a large project using "clang-cl -fallback -fsanitize=address".
//
// Check both -GS and -GS- builds:
// RUN: cl -c %s -Fo%t.obj
// RUN: %clangxx_asan -o %t.exe %s %t.obj
// RUN: %run %t.exe
//
// RUN: cl -GS- -c %s -Fo%t.obj
// RUN: %clangxx_asan -o %t.exe %s %t.obj
// RUN: %run %t.exe

#include <windows.h>
#include <assert.h>
#include <stdio.h>

// Should just "#include <sanitizer/asan_interface.h>" when C++ exceptions are
// supported and we don't need to use CL.
extern "C" bool __asan_address_is_poisoned(void *p);

void ThrowAndCatch();

#if !defined(__clang__)
__declspec(noinline)
void Throw() {
  int local, zero = 0;
  fprintf(stderr, "Throw:  %p\n", &local);
  local = 5 / zero;
}

__declspec(noinline)
void ThrowAndCatch() {
  int local;
  __try {
    Throw();
  } __except(EXCEPTION_EXECUTE_HANDLER) {
    fprintf(stderr, "__except:  %p\n", &local);
  }
}
#else

int main() {
  char x[32];
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  assert(__asan_address_is_poisoned(x + 32));
  ThrowAndCatch();
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  // FIXME: Invert this assertion once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  assert(!__asan_address_is_poisoned(x + 32));
}
#endif
