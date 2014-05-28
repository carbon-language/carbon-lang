// RUN: %clangxx_asan -O0 %s -o %t
// RUN: env ASAN_OPTIONS=malloc_context_size=0:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os
// RUN: env ASAN_OPTIONS=malloc_context_size=0:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os
// RUN: env ASAN_OPTIONS=malloc_context_size=1:fast_unwind_on_malloc=0 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os
// RUN: env ASAN_OPTIONS=malloc_context_size=1:fast_unwind_on_malloc=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os
// RUN: env ASAN_OPTIONS=malloc_context_size=2 not %run %t 2>&1 | FileCheck %s --check-prefix=TWO

int main() {
  char *x = new char[20];
  delete[] x;
  return x[0];
  // We need to keep duplicate lines with different 'CHECK-%os' prefixes,
  // otherwise FileCheck barks on missing 'CHECK-%os' before 'CHECK-%os-NEXT'.

  // CHECK-Linux: freed by thread T{{.*}} here:
  // CHECK-Linux-NEXT: #0 0x{{.*}} in operator delete[]
  // CHECK-Darwin: freed by thread T{{.*}} here:
  // CHECK-Darwin-NEXT: #0 0x{{.*}} in wrap__ZdaPv
  // CHECK-Windows: freed by thread T{{.*}} here:
  // CHECK-Windows-NEXT: #0 0x{{.*}} in operator delete[]
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK-Linux: previously allocated by thread T{{.*}} here:
  // CHECK-Linux-NEXT: #0 0x{{.*}} in operator new[]
  // CHECK-Darwin: previously allocated by thread T{{.*}} here:
  // CHECK-Darwin-NEXT: #0 0x{{.*}} in wrap__Znam
  // CHECK-Windows: previously allocated by thread T{{.*}} here:
  // CHECK-Windows-NEXT: #0 0x{{.*}} in operator new[]
  // CHECK-NOT: #1 0x{{.*}}

  // CHECK: SUMMARY: AddressSanitizer: heap-use-after-free

  // TWO: previously allocated by thread T{{.*}} here:
  // TWO-NEXT: #0 0x{{.*}}
  // TWO-NEXT: #1 0x{{.*}} in main {{.*}}malloc_context_size.cc
  // TWO: SUMMARY: AddressSanitizer: heap-use-after-free
}
