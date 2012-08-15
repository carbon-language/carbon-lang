// RUN: %clang_asan -m64 -O2 %s -o %t
// RUN: %t 2>&1 | %symbolizer | c++filt > %t.output
// RUN: FileCheck %s < %t.output
// RUN: FileCheck %s --check-prefix=CHECK-%os < %t.output

__attribute__((noinline))
static void NullDeref(int *ptr) {
  ptr[10]++;
}
int main() {
  NullDeref((int*)0);
}

// CHECK: {{.*ERROR: AddressSanitizer crashed on unknown address}}
// CHECK:   {{0x0*00028 .*pc 0x.*}}
// CHECK: {{AddressSanitizer can not provide additional info.}}

// atos on Mac cannot extract the symbol name correctly.
// CHECK-Linux: {{    #0 0x.* in NullDeref.*null_deref.cc:8}}
// CHECK-Darwin: {{    #0 0x.* in .*NullDeref.*null_deref.cc:8}}

// CHECK: {{    #1 0x.* in main.*null_deref.cc:11}}
