/*===---- arm_acle.h - ARM Non-Neon intrinsics -----------------------------===
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

#ifndef __ARM_ACLE_H
#define __ARM_ACLE_H

#ifndef __ARM_ACLE
#error "ACLE intrinsics support not enabled."
#endif

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Miscellaneous data-processing intrinsics */

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __clz(uint32_t t) {
  return __builtin_clz(t);
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __clzl(unsigned long t) {
  return __builtin_clzl(t);
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __clzll(uint64_t t) {
#if __SIZEOF_LONG_LONG__ == 8
  return __builtin_clzll(t);
#else
  return __builtin_clzl(t);
#endif
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __rev(uint32_t t) {
  return __builtin_bswap32(t);
}

static __inline__ unsigned long __attribute__((always_inline, nodebug))
  __revl(unsigned long t) {
#if __SIZEOF_LONG__ == 4
  return __builtin_bswap32(t);
#else
  return __builtin_bswap64(t);
#endif
}

static __inline__ uint64_t __attribute__((always_inline, nodebug))
  __revll(uint64_t t) {
  return __builtin_bswap64(t);
}


/*
 * Saturating intrinsics
 *
 * FIXME: Change guard to their corrosponding __ARM_FEATURE flag when Q flag
 * intrinsics are implemented and the flag is enabled.
 */
#if __ARM_32BIT_STATE
#define __ssat(x, y) __builtin_arm_ssat(x, y)
#define __usat(x, y) __builtin_arm_usat(x, y)

static __inline__ int32_t __attribute__((always_inline, nodebug))
  __qadd(int32_t t, int32_t v) {
  return __builtin_arm_qadd(t, v);
}

static __inline__ int32_t __attribute__((always_inline, nodebug))
  __qsub(int32_t t, int32_t v) {
  return __builtin_arm_qsub(t, v);
}

static __inline__ int32_t __attribute__((always_inline, nodebug))
__qdbl(int32_t t) {
  return __builtin_arm_qadd(t, t);
}
#endif

/* CRC32 intrinsics */
#if __ARM_FEATURE_CRC32
static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32b(uint32_t a, uint8_t b) {
  return __builtin_arm_crc32b(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32h(uint32_t a, uint16_t b) {
  return __builtin_arm_crc32h(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32w(uint32_t a, uint32_t b) {
  return __builtin_arm_crc32w(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32d(uint32_t a, uint64_t b) {
  return __builtin_arm_crc32d(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32cb(uint32_t a, uint8_t b) {
  return __builtin_arm_crc32cb(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32ch(uint32_t a, uint16_t b) {
  return __builtin_arm_crc32ch(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32cw(uint32_t a, uint32_t b) {
  return __builtin_arm_crc32cw(a, b);
}

static __inline__ uint32_t __attribute__((always_inline, nodebug))
  __crc32cd(uint32_t a, uint64_t b) {
  return __builtin_arm_crc32cd(a, b);
}
#endif

#if defined(__cplusplus)
}
#endif

#endif /* __ARM_ACLE_H */
