// Test that verifies TSan runtime doesn't contain compiler-emitted
// memcpy/memmove calls. It builds the binary with TSan and passes it to
// check_memcpy.sh script.

// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %S/check_memcpy.sh %t

int main() {
  return 0;
}

