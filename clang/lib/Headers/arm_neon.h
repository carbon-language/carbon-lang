/*===---- arm_neon.h - NEON intrinsics --------------------------------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __ARM_NEON_H
#define __ARM_NEON_H

#ifndef __ARM_NEON__
#error "NEON support not enabled"
#endif

// NEON document appears to be specified in terms of stdint types.
#include <stdint.h>

// Define some NEON-specific scalar types for floats and polynomials.
typedef float float32_t;
typedef uint8_t poly8_t;
typedef uint16_t poly16_t;

typedef __attribute__(( __vector_size__(8) ))  int8_t __neon_int8x8_t;
typedef __attribute__(( __vector_size__(16) )) int8_t __neon_int8x16_t;
typedef __attribute__(( __vector_size__(8) ))  int16_t __neon_int16x4_t;
typedef __attribute__(( __vector_size__(16) )) int16_t __neon_int16x8_t;
typedef __attribute__(( __vector_size__(8) ))  int32_t __neon_int32x2_t;
typedef __attribute__(( __vector_size__(16) )) int32_t __neon_int32x4_t;
typedef __attribute__(( __vector_size__(8) ))  int64_t __neon_int64x1_t;
typedef __attribute__(( __vector_size__(16) )) int64_t __neon_int64x2_t;
typedef __attribute__(( __vector_size__(8) ))  uint8_t __neon_uint8x8_t;
typedef __attribute__(( __vector_size__(16) )) uint8_t __neon_uint8x16_t;
typedef __attribute__(( __vector_size__(8) ))  uint16_t __neon_uint16x4_t;
typedef __attribute__(( __vector_size__(16) )) uint16_t __neon_uint16x8_t;
typedef __attribute__(( __vector_size__(8) ))  uint32_t __neon_uint32x2_t;
typedef __attribute__(( __vector_size__(16) )) uint32_t __neon_uint32x4_t;
typedef __attribute__(( __vector_size__(8) ))  uint64_t __neon_uint64x1_t;
typedef __attribute__(( __vector_size__(16) )) uint64_t __neon_uint64x2_t;
typedef __attribute__(( __vector_size__(8) ))  uint16_t __neon_float16x4_t;
typedef __attribute__(( __vector_size__(16) )) uint16_t __neon_float16x8_t;
typedef __attribute__(( __vector_size__(8) ))  float32_t __neon_float32x2_t;
typedef __attribute__(( __vector_size__(16) )) float32_t __neon_float32x4_t;
typedef __attribute__(( __vector_size__(8) ))  poly8_t __neon_poly8x8_t;
typedef __attribute__(( __vector_size__(16) )) poly8_t __neon_poly8x16_t;
typedef __attribute__(( __vector_size__(8) ))  poly16_t __neon_poly16x4_t;
typedef __attribute__(( __vector_size__(16) )) poly16_t __neon_poly16x8_t;

typedef struct __int8x8_t {
  __neon_int8x8_t val;
} int8x8_t;

typedef struct __int8x16_t {
  __neon_int8x16_t val;
} int8x16_t;

typedef struct __int16x4_t {
  __neon_int16x4_t val;
} int16x4_t;

typedef struct __int16x8_t {
  __neon_int16x8_t val;
} int16x8_t;

typedef struct __int32x2_t {
  __neon_int32x2_t val;
} int32x2_t;

typedef struct __int32x4_t {
  __neon_int32x4_t val;
} int32x4_t;

typedef struct __int64x1_t {
  __neon_int64x1_t val;
} int64x1_t;

typedef struct __int64x2_t {
  __neon_int64x2_t val;
} int64x2_t;

typedef struct __uint8x8_t {
  __neon_uint8x8_t val;
} uint8x8_t;

typedef struct __uint8x16_t {
  __neon_uint8x16_t val;
} uint8x16_t;

typedef struct __uint16x4_t {
  __neon_uint16x4_t val;
} uint16x4_t;

typedef struct __uint16x8_t {
  __neon_uint16x8_t val;
} uint16x8_t;

typedef struct __uint32x2_t {
  __neon_uint32x2_t val;
} uint32x2_t;

typedef struct __uint32x4_t {
  __neon_uint32x4_t val;
} uint32x4_t;

typedef struct __uint64x1_t {
  __neon_uint64x1_t val;
} uint64x1_t;

typedef struct __uint64x2_t {
  __neon_uint64x2_t val;
} uint64x2_t;

typedef struct __float16x4_t {
  __neon_float16x4_t val;
} float16x4_t;

typedef struct __float16x8_t {
  __neon_float16x8_t val;
} float16x8_t;

typedef struct __float32x2_t {
  __neon_float32x2_t val;
} float32x2_t;

typedef struct __float32x4_t {
  __neon_float32x4_t val;
} float32x4_t;

typedef struct __poly8x8_t {
  __neon_poly8x8_t val;
} poly8x8_t;

typedef struct __poly8x16_t {
  __neon_poly8x16_t val;
} poly8x16_t;

typedef struct __poly16x4_t {
  __neon_poly16x4_t val;
} poly16x4_t;

typedef struct __poly16x8_t {
  __neon_poly16x8_t val;
} poly16x8_t;

// FIXME: write tool to stamp out the structure-of-array types, possibly gen this whole file.

// Intrinsics, per ARM document DUI0348B
#define _ATTRS_ai __attribute__((__always_inline__))

static _ATTRS_ai int8x8_t vadd_s8(int8x8_t a, int8x8_t b) { return (int8x8_t){a.val + b.val}; }
static _ATTRS_ai int16x4_t vadd_s16(int16x4_t a, int16x4_t b) { return (int16x4_t){a.val + b.val}; }
static _ATTRS_ai int32x2_t vadd_s32(int32x2_t a, int32x2_t b) { return (int32x2_t){a.val + b.val}; }
static _ATTRS_ai int64x1_t vadd_s64(int64x1_t a, int64x1_t b) { return (int64x1_t){a.val + b.val}; }
static _ATTRS_ai float32x2_t vadd_f32(float32x2_t a, float32x2_t b) { return (float32x2_t){a.val + b.val}; }
static _ATTRS_ai uint8x8_t vadd_u8(uint8x8_t a, uint8x8_t b) { return (uint8x8_t){a.val + b.val}; }
static _ATTRS_ai uint16x4_t vadd_u16(uint16x4_t a, uint16x4_t b) { return (uint16x4_t){a.val + b.val}; }
static _ATTRS_ai uint32x2_t vadd_u32(uint32x2_t a, uint32x2_t b) { return (uint32x2_t){a.val + b.val}; }
static _ATTRS_ai uint64x1_t vadd_u64(uint64x1_t a, uint64x1_t b) { return (uint64x1_t){a.val + b.val}; }
static _ATTRS_ai int8x16_t vaddq_s8(int8x16_t a, int8x16_t b) { return (int8x16_t){a.val + b.val}; }
static _ATTRS_ai int16x8_t vaddq_s16(int16x8_t a, int16x8_t b) { return (int16x8_t){a.val + b.val}; }
static _ATTRS_ai int32x4_t vaddq_s32(int32x4_t a, int32x4_t b) { return (int32x4_t){a.val + b.val}; }
static _ATTRS_ai int64x2_t vaddq_s64(int64x2_t a, int64x2_t b) { return (int64x2_t){a.val + b.val}; }
static _ATTRS_ai float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return (float32x4_t){a.val + b.val}; }
static _ATTRS_ai uint8x16_t vaddq_u8(uint8x16_t a, uint8x16_t b) { return (uint8x16_t){a.val + b.val}; }
static _ATTRS_ai uint16x8_t vaddq_u16(uint16x8_t a, uint16x8_t b) { return (uint16x8_t){a.val + b.val}; }
static _ATTRS_ai uint32x4_t vaddq_u32(uint32x4_t a, uint32x4_t b) { return (uint32x4_t){a.val + b.val}; }
static _ATTRS_ai uint64x2_t vaddq_u64(uint64x2_t a, uint64x2_t b) { return (uint64x2_t){a.val + b.val}; }

// add
// long add
// wide add
// halving add
// rounding halving add
// saturating add
// add high half
// rounding add high half

// multiply
// multiple accumulate
// multiple accumulate long
// multiple subtract
// multiple subtract long
// saturating doubling multiply high 
// saturating rounding doubling multiply high 
// saturating doubling multiply accumulate long 
// saturating doubling multiply subtract long 
// long multiply
// saturating doubling long multiply

// subtract
// long subtract
// wide subtract
// saturating subtract
// halving subtract
// subtract high half
// rounding subtract high half

// compare eq
// compare ge
// compare le
// compare gt
// compare lt
// compare abs ge
// compare abs le
// compare abs gt
// compare abs lt
// test bits

// abs diff
// abs diff long
// abs diff accumulate
// abs diff accumulate long

// max
// min

// pairwise add
// long pairwise add
// long pairwise add accumulate
// pairwise max
// pairwise min

// recip
// recip sqrt

// shl by vec
// saturating shl by vec
// rounding shl by vec
// saturating rounding shl by vec

// shr by constant
// shl by constant
// rounding shr by constant
// shr by constant and accumulate
// rounding shr by constant and accumulate
// saturating shl by constant
// s->u saturating shl by constant
// narrowing saturating shr by constant
// s->u narrowing saturating shr by constant
// s->u rounding narrowing saturating shr by constant
// narrowing saturating shr by constant
// rounding narrowing shr by constant
// rounding narrowing saturating shr by constant
// widening shl by constant

// shr and insert
// shl and insert

// loads and stores, single vector
// loads and stores, lane
// loads, dupe

// loads and stores, arrays

// vget,vgetq lane
// vset, vsetq lane

// vcreate
// vdup, vdupq
// vmov, vmovq
// vdup_lane, vdupq_lane
// vcombine
// vget_high, vget_low

// vcvt {u,s} <-> f, f <-> f16
// narrow
// long move (unpack)
// saturating narrow
// saturating narrow s->u

// table lookup
// extended table lookup

// mla with scalar
// widening mla with scalar
// widening saturating doubling mla with scalar
// mls with scalar
// widening mls with scalar
// widening saturating doubling mls with scalar
// mul by scalar
// long mul with scalar
// long mul by scalar
// saturating doubling long mul with scalar
// saturating doubling long mul by scalar
// saturating doubling mul high with scalar
// saturating doubling mul high by scalar
// saturating rounding doubling mul high with scalar
// saturating rounding doubling mul high by scalar
// mla with scalar
// widening mla with sclar
// widening saturating doubling mla with scalar
// mls with scalar
// widening mls with scalar
// widening saturating doubling mls with scalar

// extract

// endian swap (vrev)

// abs
// saturating abs
// negate
// saturating negate
// count leading signs
// count leading zeroes
// popcount

// recip_est
// recip_sqrt_est

// not
// and
// or
// xor
// andn
// orn
// bitselect

// transpose elts
// interleave elts
// deinterleave elts

// vreinterpret


#endif /* __ARM_NEON_H */
