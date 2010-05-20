// RUN: %llvmgcc -mssse3 -S -o - %s | llc -mtriple=x86_64-apple-darwin | FileCheck %s
// XFAIL: *
// XTARGET: x86,i386,i686

#include <tmmintrin.h>

int main ()
{
#if defined( __SSSE3__ )

#define vec_rld_epi16( _a, _i )  ({ vSInt16 _t = _a; _t = _mm_alignr_epi8( _t, _t, _i ); /*return*/ _t; })
  typedef int16_t     vSInt16         __attribute__ ((__vector_size__ (16)));

  short   dtbl[] = {1,2,3,4,5,6,7,8};
  vSInt16 *vdtbl = (vSInt16*) dtbl;

  vSInt16 v0;
  v0 = *vdtbl;
  // CHECK: pshufd  $57
  v0 = vec_rld_epi16( v0, 4 );

  return 0;
#endif
}
