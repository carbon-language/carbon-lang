// RUN: %clangxx_asan -O0 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O1 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O2 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out
// RUN: %clangxx_asan -O3 %s -o %t && not %t 2>%t.out
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-%os < %t.out

__attribute__((noinline))
static void NullDeref(int *ptr) {
  // CHECK: ERROR: AddressSanitizer: SEGV on unknown address
  // CHECK:   {{0x0*00028 .*pc 0x.*}}
  // CHECK: {{AddressSanitizer can not provide additional info.}}
  ptr[10]++;  // BOOM
  // atos on Mac cannot extract the symbol name correctly.
  // CHECK-Linux: {{    #0 0x.* in NullDeref.*null_deref.cc:}}[[@LINE-2]]
  // CHECK-Darwin: {{    #0 0x.* in .*NullDeref.*null_deref.cc:}}[[@LINE-3]]
}
int main() {
  NullDeref((int*)0);
  // CHECK: {{    #1 0x.* in main.*null_deref.cc:}}[[@LINE-1]]
}
