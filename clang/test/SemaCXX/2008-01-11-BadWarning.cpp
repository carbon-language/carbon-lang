// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s
// rdar://5683899
void** f(void **Buckets, unsigned NumBuckets) {
  return Buckets + NumBuckets;
}
