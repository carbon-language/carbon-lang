// RUN: %clangxx_msan -m64 -O0 %s -o %t && %t
// RUN: %clangxx_msan -DPOSITIVE -m64 -O0 %s -o %t && not %t 2>&1 | FileCheck %s

#include <emmintrin.h>

int to_int(double v) {
  __m128d t = _mm_set_sd(v);
  int x = _mm_cvtsd_si32(t);
  return x;
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #{{.*}} in to_int{{.*}}vector_cvt.cc:[[@LINE-4]]
}

int main() {
#ifdef POSITIVE
  double v;
#else
  double v = 1.1;
#endif
  double* volatile p = &v;
  int x = to_int(*p);
  return !x;
}
