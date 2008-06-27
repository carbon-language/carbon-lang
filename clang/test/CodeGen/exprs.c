// RUN: clang %s -emit-llvm -o -

// PR1895
// sizeof function
int zxcv(void);
int x=sizeof(zxcv);
int y=__alignof__(zxcv);


void *test(int *i) {
 short a = 1;
 i += a;
 i + a;
 a + i;
}

_Bool test2b; 
int test2() {if (test2b);}

// PR1921
int test3() {
  const unsigned char *bp;
  bp -= (short)1;
}

// PR2080 - sizeof void
int t1 = sizeof(void);
int t2 = __alignof__(void);
void test4() {
  t1 = sizeof(void);
  t2 = __alignof__(void);
  
  t1 = sizeof(test4());
  t2 = __alignof__(test4());
}

// 'const float' promotes to double in varargs.
int test5(const float x, float float_number) {
  return __builtin_isless(x, float_number);
}

