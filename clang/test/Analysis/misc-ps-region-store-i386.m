// RUN: clang-cc -triple i386-apple-darwin9 -analyze -checker-cfref --analyzer-store=region --verify -fblocks %s

typedef struct _BStruct { void *grue; } BStruct;
void testB_aux(void *ptr);
void testB(BStruct *b) {
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    int __gruev2__ = *__gruep__;
    if (__gruev__ != __gruev2__) {
      int *p = 0;
      *p = 0xDEADBEEF;
    }
    
    testB_aux(__gruep__);
  }
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    int __gruev2__ = *__gruep__;
    if (__gruev__ != __gruev2__) {
      int *p = 0;
      *p = 0xDEADBEEF;
    }
    
    if (~0 != __gruev__) {}
  }
}

