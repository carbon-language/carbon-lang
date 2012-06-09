//===-- tsan_defs.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#ifndef TSAN_DEFS_H
#define TSAN_DEFS_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_stat.h"

#ifndef TSAN_DEBUG
#define TSAN_DEBUG 0
#endif  // TSAN_DEBUG

namespace __tsan {

const int kTidBits = 13;
const unsigned kMaxTid = 1 << kTidBits;
const unsigned kMaxTidInClock = kMaxTid * 2;  // This includes msb 'freed' bit.
const int kClkBits = 43;

#ifdef TSAN_SHADOW_COUNT
# if TSAN_SHADOW_COUNT == 2 \
  || TSAN_SHADOW_COUNT == 4 || TSAN_SHADOW_COUNT == 8
const unsigned kShadowCnt = TSAN_SHADOW_COUNT;
# else
#   error "TSAN_SHADOW_COUNT must be one of 2,4,8"
# endif
#else
// Count of shadow values in a shadow cell.
const unsigned kShadowCnt = 8;
#endif

// That many user bytes are mapped onto a single shadow cell.
const unsigned kShadowCell = 8;

// Size of a single shadow value (u64).
const unsigned kShadowSize = 8;

#if defined(TSAN_COLLECT_STATS) && TSAN_COLLECT_STATS
const bool kCollectStats = true;
#else
const bool kCollectStats = false;
#endif

#if TSAN_DEBUG
#define DCHECK(a)       CHECK(a)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#else
#define DCHECK(a)
#define DCHECK_EQ(a, b)
#define DCHECK_NE(a, b)
#define DCHECK_LT(a, b)
#define DCHECK_LE(a, b)
#define DCHECK_GT(a, b)
#define DCHECK_GE(a, b)
#endif

// The following "build consistency" machinery ensures that all source files
// are built in the same configuration. Inconsistent builds lead to
// hard to debug crashes.
#if TSAN_DEBUG
void build_consistency_debug();
#else
void build_consistency_release();
#endif

#if TSAN_COLLECT_STATS
void build_consistency_stats();
#else
void build_consistency_nostats();
#endif

#if TSAN_SHADOW_COUNT == 1
void build_consistency_shadow1();
#elif TSAN_SHADOW_COUNT == 2
void build_consistency_shadow2();
#elif TSAN_SHADOW_COUNT == 4
void build_consistency_shadow4();
#else
void build_consistency_shadow8();
#endif

static inline void USED build_consistency() {
#if TSAN_DEBUG
  build_consistency_debug();
#else
  build_consistency_release();
#endif
#if TSAN_COLLECT_STATS
  build_consistency_stats();
#else
  build_consistency_nostats();
#endif
#if TSAN_SHADOW_COUNT == 1
  build_consistency_shadow1();
#elif TSAN_SHADOW_COUNT == 2
  build_consistency_shadow2();
#elif TSAN_SHADOW_COUNT == 4
  build_consistency_shadow4();
#else
  build_consistency_shadow8();
#endif
}

template<typename T>
T min(T a, T b) {
  return a < b ? a : b;
}

template<typename T>
T max(T a, T b) {
  return a > b ? a : b;
}

template<typename T>
T RoundUp(T p, int align) {
  DCHECK_EQ(align & (align - 1), 0);
  return (T)(((u64)p + align - 1) & ~(align - 1));
}

void real_memset(void *ptr, int c, uptr size);
void real_memcpy(void *dst, const void *src, uptr size);
int internal_memcmp(const void *s1, const void *s2, uptr size);
int internal_strncmp(const char *s1, const char *s2, uptr size);
void internal_strcpy(char *s1, const char *s2);
const char *internal_strstr(const char *where, const char *what);
const char *internal_strchr(const char *where, char what);

struct MD5Hash {
  u64 hash[2];
  bool operator==(const MD5Hash &other) const {
    return hash[0] == other.hash[0] && hash[1] == other.hash[1];
  }
};

MD5Hash md5_hash(const void *data, uptr size);

struct ThreadState;
struct ThreadContext;
struct Context;
struct ReportStack;
class ReportDesc;
class RegionAlloc;
class StackTrace;

}  // namespace __tsan

#endif  // TSAN_DEFS_H
