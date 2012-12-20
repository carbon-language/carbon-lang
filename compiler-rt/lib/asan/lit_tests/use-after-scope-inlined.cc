// Test with "-O2" only to make sure inlining (leading to use-after-scope)
// happens. "always_inline" is not enough, as Clang doesn't emit
// llvm.lifetime intrinsics at -O0.
//
// RUN: %clangxx_asan -m64 -O2 -fsanitize=use-after-scope %s -o %t && \
// RUN:     %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -m32 -O2 -fsanitize=use-after-scope %s -o %t && \
// RUN:     %t 2>&1 | %symbolize | FileCheck %s

int *arr;

__attribute__((always_inline))
void inlined(int arg) {
  int x[5];
  for (int i = 0; i < arg; i++) x[i] = i;
  arr = x;
}

int main(int argc, char *argv[]) {
  inlined(argc);
  return arr[argc - 1];  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK: READ of size 4 at 0x{{.*}} thread T0
  // CHECK:   #0 0x{{.*}} in {{_?}}main
  // CHECK:      {{.*}}use-after-scope-inlined.cc:[[@LINE-4]]
  // CHECK: Address 0x{{.*}} is located at offset
  // CHECK:      [[OFFSET:[^ ]*]] in frame <main> of T0{{.*}}:
  // CHECK:   {{\[}}[[OFFSET]], {{.*}}) 'x.i'
}
