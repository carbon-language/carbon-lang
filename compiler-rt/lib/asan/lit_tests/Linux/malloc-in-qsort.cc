// RUN: %clangxx_asan -O2 %s -o %t
// RUN: ASAN_OPTIONS=fast_unwind_on_malloc=1 %t 2>&1 | %symbolize | FileCheck %s --check-prefix=CHECK-FAST
// RUN: ASAN_OPTIONS=fast_unwind_on_malloc=0 %t 2>&1 | %symbolize | FileCheck %s --check-prefix=CHECK-SLOW

// Test how well we unwind in presence of qsort in the stack
// (i.e. if we can unwind through a function compiled w/o frame pointers).
// https://code.google.com/p/address-sanitizer/issues/detail?id=137
#include <stdlib.h>
#include <stdio.h>

int *GlobalPtr;

extern "C" {
int QsortCallback(const void *a, const void *b) {
  char *x = (char*)a;
  char *y = (char*)b;
  printf("Calling QsortCallback\n");
  GlobalPtr = new int[10];
  return (int)*x - (int)*y;
}

__attribute__((noinline))
void MyQsort(char *a, size_t size) {
  printf("Calling qsort\n");
  qsort(a, size, sizeof(char), QsortCallback);
  printf("Done\n");  // Avoid tail call.
}
}  // extern "C"

int main() {
  char a[2] = {1, 2};
  MyQsort(a, 2);
  return GlobalPtr[10];
}

// Fast unwind: can not unwind through qsort.
// FIXME: this test does not properly work with slow unwind yet.

// CHECK-FAST: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK-FAST: is located 0 bytes to the right
// CHECK-FAST: #0{{.*}}operator new
// CHECK-FAST-NEXT: #1{{.*}}QsortCallback
// CHECK-FAST-NOT: MyQsort
//
// CHECK-SLOW: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK-SLOW: is located 0 bytes to the right
// CHECK-SLOW: #0{{.*}}operator new
// CHECK-SLOW-NEXT: #1{{.*}}QsortCallback
// CHECK-SLOW: #{{.*}}MyQsort
// CHECK-SLOW-NEXT: #{{.*}}main
