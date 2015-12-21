// Check if tsan work with PIE binaries.
// RUN: %clang_tsan %s -pie -fpic -o %t && %run %t

// Some kernels might map PIE segments outside the current segment
// mapping defined for x86 [1].
// [1] https://git.kernel.org/linus/d1fd836dcf00d2028c700c7e44d2c23404062c90

// UNSUPPORTED: x86

int main(void) {
  return 0;
}
