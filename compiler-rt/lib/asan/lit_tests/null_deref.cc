// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m64 -O1 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m64 -O2 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m64 -O3 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O0 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O1 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O2 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -m32 -O3 %s -o %t && %t 2>&1 | %symbolize > %t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out

__attribute__((noinline))
static void NullDeref(int *ptr) {
  ptr[10]++;
}
int main() {
  NullDeref((int*)0);
}

// CHECK: ERROR: AddressSanitizer: SEGV on unknown address
// CHECK:   {{0x0*00028 .*pc 0x.*}}
// CHECK: {{AddressSanitizer can not provide additional info.}}

// atos on Mac cannot extract the symbol name correctly.
// CHECK-Linux: {{    #0 0x.* in NullDeref.*null_deref.cc:20}}
// CHECK-Darwin: {{    #0 0x.* in .*NullDeref.*null_deref.cc:20}}

// CHECK: {{    #1 0x.* in _?main.*null_deref.cc:23}}
