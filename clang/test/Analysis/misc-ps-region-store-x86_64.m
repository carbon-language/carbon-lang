// RUN: clang-cc -triple x86_64-apple-darwin9 -analyze -checker-cfref --analyzer-store=region --verify -fblocks %s

// This test case appears in misc-ps-region-store-i386.m, but fails under x86_64.
// The reason is that 'int' is smaller than a pointer on a 64-bit architecture,
// and we aren't reasoning yet about just the first 32-bits of the pointer.
typedef struct _BStruct { void *grue; } BStruct;
void testB_aux(void *ptr);
void testB(BStruct *b) {
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    int __gruev2__ = *__gruep__;
    if (__gruev__ != __gruev2__) {
      int *p = 0;
      *p = 0xDEADBEEF; // no-warning
    }
    
    testB_aux(__gruep__);
  }
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    int __gruev2__ = *__gruep__;
    if (__gruev__ != __gruev2__) {
      int *p = 0;
      *p = 0xDEADBEEF; // expected-warning{{null}}
    }
    
    if (~0 != __gruev__) {}
  }
}
