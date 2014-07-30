// RUN: %clangxx_asan -Xclang -fsized-deallocation -O0 %s -o %t
// RUN:                                         not %run %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=new_delete_type_mismatch=1 not %run %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=new_delete_type_mismatch=0     %run %t

// Sized-delete is implemented with a weak delete() definition.
// Weak symbols are kind of broken on Android.
// XFAIL: android

#include <new>
#include <stdio.h>

inline void break_optimization(void *arg) {
  __asm__ __volatile__("" : : "r" (arg) : "memory");
}

struct S12 {
  int a, b, c;
};

struct S20 {
  int a, b, c, d, e;
};

void Del12(S12 *x) {
  break_optimization(x);
  delete x;
}
void Del12NoThrow(S12 *x) {
  break_optimization(x);
  operator delete(x, std::nothrow);
}
void Del12Ar(S12 *x) {
  break_optimization(x);
  delete [] x;
}
void Del12ArNoThrow(S12 *x) {
  break_optimization(x);
  operator delete[](x, std::nothrow);
}

int main() {
  // These are correct.
  Del12(new S12);
  Del12NoThrow(new S12);
  Del12Ar(new S12[100]);
  Del12ArNoThrow(new S12[100]);

  // Here we pass wrong type of pointer to delete,
  // but [] and nothrow variants of delete are not sized.
  Del12Ar(reinterpret_cast<S12*>(new S20[100]));
  Del12NoThrow(reinterpret_cast<S12*>(new S20));
  Del12ArNoThrow(reinterpret_cast<S12*>(new S20[100]));
  fprintf(stderr, "OK SO FAR\n");
  // CHECK: OK SO FAR
  // Here asan should bark as we are passing a wrong type of pointer
  // to sized delete.
  Del12(reinterpret_cast<S12*>(new S20));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  size of the allocated type:   20 bytes;
  // CHECK:  size of the deallocated type: 12 bytes.
  // CHECK: is located 0 bytes inside of 20-byte region
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch
}
