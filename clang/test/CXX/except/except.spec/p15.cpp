// RUN: %clang_cc1 -std=c++0x -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// Deallocation functions are implicitly noexcept.
// Thus, explicit specs aren't allowed to conflict.

void f() {
  // Force implicit declaration of delete.
  delete new int;
  delete[] new int[1];
}

void operator delete(void*) noexcept;
void operator delete[](void*) noexcept;

// Same goes for explicit declarations.
void operator delete(void*, float);
void operator delete(void*, float) noexcept;

void operator delete[](void*, float);
void operator delete[](void*, float) noexcept;

// But explicit specs stay.
void operator delete(void*, double) throw(int); // expected-note {{previous}}
void operator delete(void*, double) noexcept; // expected-error {{does not match}}
