// RUN: not %llvmgcc -mssse3 -S -o /dev/null %s |& grep "error: mask must be an immediate"
// XFAIL: *
// XTARGET: x86,i386,i686

#include <tmmintrin.h>

extern int i;

int main ()
{
#if defined( __SSSE3__ )

  typedef int16_t     vSInt16         __attribute__ ((__vector_size__ (16)));

  short   dtbl[] = {1,2,3,4,5,6,7,8};
  vSInt16 *vdtbl = (vSInt16*) dtbl;

  vSInt16 v0;
  v0 = *vdtbl;
  v0 = _mm_alignr_epi8(v0, v0, i);

  return 0;
#endif
}
