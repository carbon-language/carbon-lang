// RUN: %clangxx_asan -O %s -o %t && not %t 2>&1 | FileCheck %s
// Test crash due to __sanitizer_annotate_contiguous_container.

extern "C" {
void __sanitizer_annotate_contiguous_container(const void *beg, const void *end,
                                               const void *old_mid,
                                               const void *new_mid);
}  // extern "C"

int main(int argc, char **argv) {
  long t[100];
  __sanitizer_annotate_contiguous_container(&t[0], &t[0] + 100, &t[0] + 100,
                                            &t[0] + 50);
  return t[60 * argc];  // Touches the poisoned memory.
}
// CHECK: AddressSanitizer: container-overflow
