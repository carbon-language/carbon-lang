// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %t 2>&1 | FileCheck %s --check-prefix=YES
// RUN: ASAN_OPTIONS=print_summary=false not %t 2>&1 | FileCheck %s --check-prefix=NO

int main() {
  char *x = new char[20];
  delete[] x;
  return x[0];
  // YES: ERROR: AddressSanitizer: heap-use-after-free
  // YES: SUMMARY: AddressSanitizer: heap-use-after-free
  // NO: ERROR: AddressSanitizer: heap-use-after-free
  // NO-NOT: SUMMARY: AddressSanitizer: heap-use-after-free
}

