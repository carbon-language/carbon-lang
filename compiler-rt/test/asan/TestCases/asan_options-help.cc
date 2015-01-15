// RUN: %clangxx_asan -O0 %s -o %t
// RUN: ASAN_OPTIONS=help=1 %run %t 2>&1 | FileCheck %s

int main() {
}

// CHECK: Available flags for AddressSanitizer:
// CHECK-DAG: handle_segv
// CHECK-DAG: check_initialization_order
