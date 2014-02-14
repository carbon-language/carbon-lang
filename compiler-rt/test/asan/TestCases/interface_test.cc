// Check that user may include ASan interface header.
// RUN: %clang_asan %s -o %t && %t
// RUN: %clang %s -o %t && %t
#include <sanitizer/asan_interface.h>

int main() {
  return 0;
}
