// RUN: %clang_cc1 -emit-llvm -march=armv7a %s 

// XFAIL: *
// XTARGET: arm

typedef struct __simd128_uint16_t
{
  __neon_uint16x8_t val;
} uint16x8_t;

void b(uint16x8_t sat, uint16x8_t luma)
{
  __asm__("vmov.16 %1, %0   \n\t"
                                           "vtrn.16 %0, %1   \n\t"
   :"=w"(luma), "=w"(sat)
   :"0"(luma)
   );

}
