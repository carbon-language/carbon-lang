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

// FIXME: probably need a 'poly' attribute or something for correct codegen to
//        disambiguate from uint16_t.
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
#define __ai static __attribute__((__always_inline__))

#define INTTYPES_WIDE(op, builtin) \
  __ai int16x8_t op##_s8(int16x8_t a, int8x8_t b) { return (int16x8_t){ builtin(a.val, b.val) }; } \
  __ai int32x4_t op##_s16(int32x4_t a, int16x4_t b) { return (int32x4_t){ builtin(a.val, b.val) }; } \
  __ai int64x2_t op##_s32(int64x2_t a, int32x2_t b) { return (int64x2_t){ builtin(a.val, b.val) }; } \
  __ai uint16x8_t op##_u8(uint16x8_t a, uint8x8_t b) { return (uint16x8_t){ builtin(a.val, b.val) }; } \
  __ai uint32x4_t op##_u16(uint32x4_t a, uint16x4_t b) { return (uint32x4_t){ builtin(a.val, b.val) }; } \
  __ai uint64x2_t op##_u32(uint64x2_t a, uint32x2_t b) { return (uint64x2_t){ builtin(a.val, b.val) }; }

#define INTTYPES_WIDENING(op, builtin) \
  __ai int16x8_t op##_s8(int8x8_t a, int8x8_t b) { return (int16x8_t){ builtin(a.val, b.val) }; } \
  __ai int32x4_t op##_s16(int16x4_t a, int16x4_t b) { return (int32x4_t){ builtin(a.val, b.val) }; } \
  __ai int64x2_t op##_s32(int32x2_t a, int32x2_t b) { return (int64x2_t){ builtin(a.val, b.val) }; } \
  __ai uint16x8_t op##_u8(uint8x8_t a, uint8x8_t b) { return (uint16x8_t){ builtin(a.val, b.val) }; } \
  __ai uint32x4_t op##_u16(uint16x4_t a, uint16x4_t b) { return (uint32x4_t){ builtin(a.val, b.val) }; } \
  __ai uint64x2_t op##_u32(uint32x2_t a, uint32x2_t b) { return (uint64x2_t){ builtin(a.val, b.val) }; }

#define INTTYPES_WIDENING_MUL(op, builtin) \
  __ai int16x8_t op##_s8(int16x8_t a, int8x8_t b, int8x8_t c) { return (int16x8_t){ builtin(a.val, b.val, c.val) }; } \
  __ai int32x4_t op##_s16(int32x4_t a, int16x4_t b, int16x4_t c) { return (int32x4_t){ builtin(a.val, b.val, c.val) }; } \
  __ai int64x2_t op##_s32(int64x2_t a, int32x2_t b, int32x2_t c) { return (int64x2_t){ builtin(a.val, b.val, c.val) }; } \
  __ai uint16x8_t op##_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) { return (uint16x8_t){ builtin(a.val, b.val, c.val) }; } \
  __ai uint32x4_t op##_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) { return (uint32x4_t){ builtin(a.val, b.val, c.val) }; } \
  __ai uint64x2_t op##_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) { return (uint64x2_t){ builtin(a.val, b.val, c.val) }; }

#define INTTYPES_NARROWING(op, builtin) \
  __ai int8x8_t op##_s16(int16x8_t a, int16x8_t b) { return (int8x8_t){ builtin(a.val, b.val) }; } \
  __ai int16x4_t op##_s32(int32x4_t a, int32x4_t b) { return (int16x4_t){ builtin(a.val, b.val) }; } \
  __ai int32x2_t op##_s64(int64x2_t a, int64x2_t b) { return (int32x2_t){ builtin(a.val, b.val) }; } \
  __ai uint8x8_t op##_u16(uint16x8_t a, uint16x8_t b) { return (uint8x8_t){ builtin(a.val, b.val) }; } \
  __ai uint16x4_t op##_u32(uint32x4_t a, uint32x4_t b) { return (uint16x4_t){ builtin(a.val, b.val) }; } \
  __ai uint32x2_t op##_u64(uint64x2_t a, uint64x2_t b) { return (uint32x2_t){ builtin(a.val, b.val) }; }

#define INTTYPES_ADD_32(op, builtin) \
  __ai int8x8_t op##_s8(int8x8_t a, int8x8_t b) { return (int8x8_t){ builtin(a.val, b.val) }; } \
  __ai int16x4_t op##_s16(int16x4_t a, int16x4_t b) { return (int16x4_t){ builtin(a.val, b.val) }; } \
  __ai int32x2_t op##_s32(int32x2_t a, int32x2_t b) { return (int32x2_t){ builtin(a.val, b.val) }; } \
  __ai uint8x8_t op##_u8(uint8x8_t a, uint8x8_t b) { return (uint8x8_t){ builtin(a.val, b.val) }; } \
  __ai uint16x4_t op##_u16(uint16x4_t a, uint16x4_t b) { return (uint16x4_t){ builtin(a.val, b.val) }; } \
  __ai uint32x2_t op##_u32(uint32x2_t a, uint32x2_t b) { return (uint32x2_t){ builtin(a.val, b.val) }; } \
  __ai int8x16_t op##q_s8(int8x16_t a, int8x16_t b) { return (int8x16_t){ builtin(a.val, b.val) }; } \
  __ai int16x8_t op##q_s16(int16x8_t a, int16x8_t b) { return (int16x8_t){ builtin(a.val, b.val) }; } \
  __ai int32x4_t op##q_s32(int32x4_t a, int32x4_t b) { return (int32x4_t){ builtin(a.val, b.val) }; } \
  __ai uint8x16_t op##q_u8(uint8x16_t a, uint8x16_t b) { return (uint8x16_t){ builtin(a.val, b.val) }; } \
  __ai uint16x8_t op##q_u16(uint16x8_t a, uint16x8_t b) { return (uint16x8_t){ builtin(a.val, b.val) }; } \
  __ai uint32x4_t op##q_u32(uint32x4_t a, uint32x4_t b) { return (uint32x4_t){ builtin(a.val, b.val) }; }

#define INTTYPES_ADD_64(op, builtin) \
  __ai int64x1_t op##_s64(int64x1_t a, int64x1_t b) { return (int64x1_t){ builtin(a.val, b.val) }; } \
  __ai uint64x1_t op##_u64(uint64x1_t a, uint64x1_t b) { return (uint64x1_t){ builtin(a.val, b.val) }; } \
  __ai int64x2_t op##q_s64(int64x2_t a, int64x2_t b) { return (int64x2_t){ builtin(a.val, b.val) }; } \
  __ai uint64x2_t op##q_u64(uint64x2_t a, uint64x2_t b) { return (uint64x2_t){ builtin(a.val, b.val) }; }

#define FLOATTYPES_CMP(op, builtin) \
  __ai uint32x2_t op##_f32(float32x2_t a, float32x2_t b) { return (uint32x2_t){ builtin(a.val, b.val) }; } \
  __ai uint32x4_t op##q_f32(float32x4_t a, float32x4_t b) { return (uint32x4_t){ builtin(a.val, b.val) }; }

#define INT_FLOAT_CMP_OP(op, cc) \
  __ai uint8x8_t op##_s8(int8x8_t a, int8x8_t b) { return (uint8x8_t){(__neon_uint8x8_t)(a.val cc b.val)}; } \
  __ai uint16x4_t op##_s16(int16x4_t a, int16x4_t b) { return (uint16x4_t){(__neon_uint16x4_t)(a.val cc b.val)}; } \
  __ai uint32x2_t op##_s32(int32x2_t a, int32x2_t b) { return (uint32x2_t){(__neon_uint32x2_t)(a.val cc b.val)}; } \
  __ai uint32x2_t op##_f32(float32x2_t a, float32x2_t b) { return (uint32x2_t){(__neon_uint32x2_t)(a.val cc b.val)}; } \
  __ai uint8x8_t op##_u8(uint8x8_t a, uint8x8_t b) { return (uint8x8_t){a.val cc b.val}; } \
  __ai uint16x4_t op##_u16(uint16x4_t a, uint16x4_t b) { return (uint16x4_t){a.val cc b.val}; } \
  __ai uint32x2_t op##_u32(uint32x2_t a, uint32x2_t b) { return (uint32x2_t){a.val cc b.val}; } \
  __ai uint8x16_t op##q_s8(int8x16_t a, int8x16_t b) { return (uint8x16_t){(__neon_uint8x16_t)(a.val cc b.val)}; } \
  __ai uint16x8_t op##q_s16(int16x8_t a, int16x8_t b) { return (uint16x8_t){(__neon_uint16x8_t)(a.val cc b.val)}; } \
  __ai uint32x4_t op##q_s32(int32x4_t a, int32x4_t b) { return (uint32x4_t){(__neon_uint32x4_t)(a.val cc b.val)}; } \
  __ai uint32x4_t op##q_f32(float32x4_t a, float32x4_t b) { return (uint32x4_t){(__neon_uint32x4_t)(a.val cc b.val)}; } \
  __ai uint8x16_t op##q_u8(uint8x16_t a, uint8x16_t b) { return (uint8x16_t){a.val cc b.val}; } \
  __ai uint16x8_t op##q_u16(uint16x8_t a, uint16x8_t b) { return (uint16x8_t){a.val cc b.val}; } \
  __ai uint32x4_t op##q_u32(uint32x4_t a, uint32x4_t b) { return (uint32x4_t){a.val cc b.val}; } 

#define INT_UNARY(op, builtin) \
  __ai int8x8_t op##_s8(int8x8_t a) { return (int8x8_t){ builtin(a.val) }; } \
  __ai int16x4_t op##_s16(int16x4_t a) { return (int16x4_t){ builtin(a.val) }; } \
  __ai int32x2_t op##_s32(int32x2_t a) { return (int32x2_t){ builtin(a.val) }; } \
  __ai int8x16_t op##q_s8(int8x16_t a) { return (int8x16_t){ builtin(a.val) }; } \
  __ai int16x8_t op##q_s16(int16x8_t a) { return (int16x8_t){ builtin(a.val) }; } \
  __ai int32x4_t op##q_s32(int32x4_t a) { return (int32x4_t){ builtin(a.val) }; }

#define FP_UNARY(op, builtin) \
  __ai float32x2_t op##_f32(float32x2_t a) { return (float32x2_t){ builtin(a.val) }; } \
  __ai float32x4_t op##q_f32(float32x4_t a) { return (float32x4_t){ builtin(a.val) }; }

#define FP_BINARY(op, builtin) \
  __ai float32x2_t op##_f32(float32x2_t a, float32x2_t b) { return (float32x2_t){ builtin(a.val, b.val) }; } \
  __ai float32x4_t op##q_f32(float32x4_t a, float32x4_t b) { return (float32x4_t){ builtin(a.val, b.val) }; }

#define INT_FP_PAIRWISE_ADD(op, builtin) \
  __ai int8x8_t op##_s8(int8x8_t a, int8x8_t b) { return (int8x8_t){ builtin(a.val, b.val) }; } \
  __ai int16x4_t op##_s16(int16x4_t a, int16x4_t b) { return (int16x4_t){ builtin(a.val, b.val) }; } \
  __ai int32x2_t op##_s32(int32x2_t a, int32x2_t b) { return (int32x2_t){ builtin(a.val, b.val) }; } \
  __ai uint8x8_t op##_u8(uint8x8_t a, uint8x8_t b) { return (uint8x8_t){ builtin(a.val, b.val) }; } \
  __ai uint16x4_t op##_u16(uint16x4_t a, uint16x4_t b) { return (uint16x4_t){ builtin(a.val, b.val) }; } \
  __ai uint32x2_t op##_u32(uint32x2_t a, uint32x2_t b) { return (uint32x2_t){ builtin(a.val, b.val) }; } \
  __ai float32x2_t op##_f32(float32x2_t a, float32x2_t b) { return (float32x2_t){ builtin(a.val, b.val) }; }

#define INT_LOGICAL_OP(op, lop) \
  __ai int8x8_t op##_s8(int8x8_t a, int8x8_t b) { return (int8x8_t){ a.val lop b.val }; } \
  __ai int16x4_t op##_s16(int16x4_t a, int16x4_t b) { return (int16x4_t){ a.val lop b.val }; } \
  __ai int32x2_t op##_s32(int32x2_t a, int32x2_t b) { return (int32x2_t){ a.val lop b.val }; } \
  __ai int64x1_t op##_s64(int64x1_t a, int64x1_t b) { return (int64x1_t){ a.val lop b.val }; } \
  __ai uint8x8_t op##_u8(uint8x8_t a, uint8x8_t b) { return (uint8x8_t){ a.val lop b.val }; } \
  __ai uint16x4_t op##_u16(uint16x4_t a, uint16x4_t b) { return (uint16x4_t){ a.val lop b.val }; } \
  __ai uint32x2_t op##_u32(uint32x2_t a, uint32x2_t b) { return (uint32x2_t){ a.val lop b.val }; } \
  __ai uint64x1_t op##_u64(uint64x1_t a, uint64x1_t b) { return (uint64x1_t){ a.val lop b.val }; } \
  __ai int8x16_t op##q_s8(int8x16_t a, int8x16_t b) { return (int8x16_t){ a.val lop b.val }; } \
  __ai int16x8_t op##q_s16(int16x8_t a, int16x8_t b) { return (int16x8_t){ a.val lop b.val }; } \
  __ai int32x4_t op##q_s32(int32x4_t a, int32x4_t b) { return (int32x4_t){ a.val lop b.val }; } \
  __ai int64x2_t op##q_s64(int64x2_t a, int64x2_t b) { return (int64x2_t){ a.val lop b.val }; } \
  __ai uint8x16_t op##q_u8(uint8x16_t a, uint8x16_t b) { return (uint8x16_t){ a.val lop b.val }; } \
  __ai uint16x8_t op##q_u16(uint16x8_t a, uint16x8_t b) { return (uint16x8_t){ a.val lop b.val }; } \
  __ai uint32x4_t op##q_u32(uint32x4_t a, uint32x4_t b) { return (uint32x4_t){ a.val lop b.val }; } \
  __ai uint64x2_t op##q_u64(uint64x2_t a, uint64x2_t b) { return (uint64x2_t){ a.val lop b.val }; }

// vector add
__ai int8x8_t vadd_s8(int8x8_t a, int8x8_t b) { return (int8x8_t){a.val + b.val}; }
__ai int16x4_t vadd_s16(int16x4_t a, int16x4_t b) { return (int16x4_t){a.val + b.val}; }
__ai int32x2_t vadd_s32(int32x2_t a, int32x2_t b) { return (int32x2_t){a.val + b.val}; }
__ai int64x1_t vadd_s64(int64x1_t a, int64x1_t b) { return (int64x1_t){a.val + b.val}; }
__ai float32x2_t vadd_f32(float32x2_t a, float32x2_t b) { return (float32x2_t){a.val + b.val}; }
__ai uint8x8_t vadd_u8(uint8x8_t a, uint8x8_t b) { return (uint8x8_t){a.val + b.val}; }
__ai uint16x4_t vadd_u16(uint16x4_t a, uint16x4_t b) { return (uint16x4_t){a.val + b.val}; }
__ai uint32x2_t vadd_u32(uint32x2_t a, uint32x2_t b) { return (uint32x2_t){a.val + b.val}; }
__ai uint64x1_t vadd_u64(uint64x1_t a, uint64x1_t b) { return (uint64x1_t){a.val + b.val}; }
__ai int8x16_t vaddq_s8(int8x16_t a, int8x16_t b) { return (int8x16_t){a.val + b.val}; }
__ai int16x8_t vaddq_s16(int16x8_t a, int16x8_t b) { return (int16x8_t){a.val + b.val}; }
__ai int32x4_t vaddq_s32(int32x4_t a, int32x4_t b) { return (int32x4_t){a.val + b.val}; }
__ai int64x2_t vaddq_s64(int64x2_t a, int64x2_t b) { return (int64x2_t){a.val + b.val}; }
__ai float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return (float32x4_t){a.val + b.val}; }
__ai uint8x16_t vaddq_u8(uint8x16_t a, uint8x16_t b) { return (uint8x16_t){a.val + b.val}; }
__ai uint16x8_t vaddq_u16(uint16x8_t a, uint16x8_t b) { return (uint16x8_t){a.val + b.val}; }
__ai uint32x4_t vaddq_u32(uint32x4_t a, uint32x4_t b) { return (uint32x4_t){a.val + b.val}; }
__ai uint64x2_t vaddq_u64(uint64x2_t a, uint64x2_t b) { return (uint64x2_t){a.val + b.val}; }

// vector long add
INTTYPES_WIDENING(vaddl, __builtin_neon_vaddl)

// vector wide add
INTTYPES_WIDE(vaddw, __builtin_neon_vaddw)

// halving add
// rounding halving add
INTTYPES_ADD_32(vhadd, __builtin_neon_vhadd)
INTTYPES_ADD_32(vrhadd, __builtin_neon_vrhadd)

// saturating add
INTTYPES_ADD_32(vqadd, __builtin_neon_vqadd)
INTTYPES_ADD_64(vqadd, __builtin_neon_vqadd)

// add high half
// rounding add high half
INTTYPES_NARROWING(vaddhn, __builtin_neon_vaddhn)
INTTYPES_NARROWING(vraddhn, __builtin_neon_vraddhn)

// multiply
// mul-poly

// multiple accumulate
// multiple subtract

// multiple accumulate long
// multiple subtract long
INTTYPES_WIDENING_MUL(vmlal, __builtin_neon_vmlal)
INTTYPES_WIDENING_MUL(vmlsl, __builtin_neon_vmlsl)

// saturating doubling multiply high 
// saturating rounding doubling multiply high 

// saturating doubling multiply accumulate long 
// saturating doubling multiply subtract long 

// long multiply
// long multiply-poly
INTTYPES_WIDENING(vmull, __builtin_neon_vmull)
__ai poly16x8_t vmull_p8(poly8x8_t a, poly8x8_t b) { return (poly16x8_t){ __builtin_neon_vmull(a.val, b.val) }; }

// saturating doubling long multiply

// subtract

// long subtract
INTTYPES_WIDENING(vsubl, __builtin_neon_vsubl)

// wide subtract
INTTYPES_WIDE(vsubw, __builtin_neon_vsubw)

// saturating subtract
INTTYPES_ADD_32(vqsub, __builtin_neon_vqsub)
INTTYPES_ADD_64(vqsub, __builtin_neon_vqsub)

// halving subtract
INTTYPES_ADD_32(vhsub, __builtin_neon_vhsub)

// subtract high half
// rounding subtract high half
INTTYPES_NARROWING(vsubhn, __builtin_neon_vsubhn)
INTTYPES_NARROWING(vrsubhn, __builtin_neon_vrsubhn)

// compare eq
// compare ge
// compare le
// compare gt
// compare lt
INT_FLOAT_CMP_OP(vceq, ==)
INT_FLOAT_CMP_OP(vcge, >=)
INT_FLOAT_CMP_OP(vcle, <=)
INT_FLOAT_CMP_OP(vcgt, >)
INT_FLOAT_CMP_OP(vclt, <)

// compare eq-poly

// compare abs ge
// compare abs le
// compare abs gt
// compare abs lt
FLOATTYPES_CMP(vcage, __builtin_neon_vcage)
FLOATTYPES_CMP(vcale, __builtin_neon_vcale)
FLOATTYPES_CMP(vcagt, __builtin_neon_vcagt)
FLOATTYPES_CMP(vcalt, __builtin_neon_vcalt)

// test bits

// abs diff
INTTYPES_ADD_32(vabd, __builtin_neon_vabd)
FP_BINARY(vabd, __builtin_neon_vabd)

// abs diff long
INTTYPES_WIDENING(vabdl, __builtin_neon_vabdl)

// abs diff accumulate
// abs diff accumulate long

// max
// min
INTTYPES_ADD_32(vmax, __builtin_neon_vmax)
FP_BINARY(vmax, __builtin_neon_vmax)
INTTYPES_ADD_32(vmin, __builtin_neon_vmin)
FP_BINARY(vmin, __builtin_neon_vmin)

// pairwise add
// pairwise max
// pairwise min
INT_FP_PAIRWISE_ADD(vpadd, __builtin_neon_vpadd)
INT_FP_PAIRWISE_ADD(vpmax, __builtin_neon_vpmax)
INT_FP_PAIRWISE_ADD(vpmin, __builtin_neon_vpmin)

// long pairwise add
// long pairwise add accumulate

// recip
// recip sqrt
FP_BINARY(vrecps, __builtin_neon_vrecps)
FP_BINARY(vrsqrts, __builtin_neon_vrsqrts)

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

// negate

// abs
// saturating abs
// saturating negate
// count leading signs
INT_UNARY(vabs, __builtin_neon_vabs)
FP_UNARY(vabs, __builtin_neon_vabs)
INT_UNARY(vqabs, __builtin_neon_vqabs)
INT_UNARY(vqneg, __builtin_neon_vqneg)
INT_UNARY(vcls, __builtin_neon_vcls)

// count leading zeroes
// popcount

// recip_est
// recip_sqrt_est

// not-poly
// not

// and
// or
// xor
// andn
// orn
INT_LOGICAL_OP(vand, &)
INT_LOGICAL_OP(vorr, |)
INT_LOGICAL_OP(veor, ^)
INT_LOGICAL_OP(vbic, &~)
INT_LOGICAL_OP(vorn, |~)

// bitselect

// transpose elts
// interleave elts
// deinterleave elts

// vreinterpret

#endif /* __ARM_NEON_H */
