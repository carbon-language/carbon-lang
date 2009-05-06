// RUN: clang-cc -triple i386-apple-darwin9 -mcpu=pentium4 -emit-llvm -o %t %s && 
// RUN: grep define %t | count 1 &&
// RUN: clang-cc -triple i386-apple-darwin9 -mcpu=pentium4 -g -emit-llvm -o %t %s && 
// RUN: grep define %t | count 1

#include <mmintrin.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int array[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
  __m64 *p = (__m64 *)array;
  
  __m64 accum = _mm_setzero_si64();
  
  for (int i=0; i<8; ++i)
    accum = _mm_add_pi32(p[i], accum);
  
  __m64 accum2 = _mm_unpackhi_pi32(accum, accum);
  accum = _mm_add_pi32(accum, accum2);
  
  int result = _mm_cvtsi64_si32(accum);
  _mm_empty();
  printf("%d\n", result );
  
  return 0;
}
