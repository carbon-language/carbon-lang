// RUN: %clangxx_asan -O0 %s -o %t
// RUN: ASAN_OPTIONS=malloc_context_size=0:fast_unwind_on_malloc=0 not %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=malloc_context_size=0:fast_unwind_on_malloc=1 not %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=malloc_context_size=1:fast_unwind_on_malloc=0 not %t 2>&1 | FileCheck %s
// RUN: ASAN_OPTIONS=malloc_context_size=1:fast_unwind_on_malloc=1 not %t 2>&1 | FileCheck %s

int main() {
  char *x = new char[20];
  delete[] x;
  return x[0];
  // CHECK: freed by thread T{{.*}} here:
  // CHECK-NEXT: #0 0x{{.*}} in operator delete[]
  // CHECK-NOT: #1 0x{{.*}}
  // CHECK: previously allocated by thread T{{.*}} here:
  // CHECK-NEXT: #0 0x{{.*}} in operator new[]
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK: SUMMARY: AddressSanitizer: heap-use-after-free
}
