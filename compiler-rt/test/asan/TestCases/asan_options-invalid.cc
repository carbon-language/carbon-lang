// RUN: %clangxx_asan -O0 %s -o %t
// RUN: ASAN_OPTIONS=invalid_option_name=10 not %run %t 2>&1 | FileCheck %s

int main() {
}

// CHECK: Unknown flag{{.*}}invalid_option_name
