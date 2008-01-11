// RUN: %llvmgcc -xc++ %s -S -o /dev/null |& not grep warning
// rdar://5683899
// XFAIL: llvmgcc4.0.1
void** f(void **Buckets, unsigned NumBuckets) {
  return Buckets + NumBuckets;
}

