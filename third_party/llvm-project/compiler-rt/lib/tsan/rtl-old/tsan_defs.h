//===-- tsan_defs.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "sanitizer_common/sanitizer_mutex.h"
#include "ubsan/ubsan_platform.h"

#ifndef TSAN_VECTORIZE
#  define TSAN_VECTORIZE __SSE4_2__
#endif

#if TSAN_VECTORIZE
// <emmintrin.h> transitively includes <stdlib.h>,
// and it's prohibited to include std headers into tsan runtime.
// So we do this dirty trick.
#  define _MM_MALLOC_H_INCLUDED
#  define __MM_MALLOC_H
#  include <emmintrin.h>
#  include <smmintrin.h>
#  define VECTOR_ALIGNED ALIGNED(16)
typedef __m128i m128;
#else
#  define VECTOR_ALIGNED
#endif

// Setup defaults for compile definitions.
#ifndef TSAN_NO_HISTORY
# define TSAN_NO_HISTORY 0
#endif

#ifndef TSAN_CONTAINS_UBSAN
# if CAN_SANITIZE_UB && !SANITIZER_GO
#  define TSAN_CONTAINS_UBSAN 1
# else
#  define TSAN_CONTAINS_UBSAN 0
# endif
#endif

namespace __tsan {

constexpr uptr kByteBits = 8;

// Thread slot ID.
enum class Sid : u8 {};
constexpr uptr kThreadSlotCount = 256;
constexpr Sid kFreeSid = static_cast<Sid>(255);

// Abstract time unit, vector clock element.
enum class Epoch : u16 {};
constexpr uptr kEpochBits = 14;
constexpr Epoch kEpochZero = static_cast<Epoch>(0);
constexpr Epoch kEpochOver = static_cast<Epoch>(1 << kEpochBits);

const int kClkBits = 42;
const unsigned kMaxTidReuse = (1 << (64 - kClkBits)) - 1;

struct ClockElem {
  u64 epoch  : kClkBits;
  u64 reused : 64 - kClkBits;  // tid reuse count
};

struct ClockBlock {
  static const uptr kSize = 512;
  static const uptr kTableSize = kSize / sizeof(u32);
  static const uptr kClockCount = kSize / sizeof(ClockElem);
  static const uptr kRefIdx = kTableSize - 1;
  static const uptr kBlockIdx = kTableSize - 2;

  union {
    u32       table[kTableSize];
    ClockElem clock[kClockCount];
  };

  ClockBlock() {
  }
};

const int kTidBits = 13;
// Reduce kMaxTid by kClockCount because one slot in ClockBlock table is
// occupied by reference counter, so total number of elements we can store
// in SyncClock is kClockCount * (kTableSize - 1).
const unsigned kMaxTid = (1 << kTidBits) - ClockBlock::kClockCount;
#if !SANITIZER_GO
const unsigned kMaxTidInClock = kMaxTid * 2;  // This includes msb 'freed' bit.
#else
const unsigned kMaxTidInClock = kMaxTid;  // Go does not track freed memory.
#endif
const uptr kShadowStackSize = 64 * 1024;

// Count of shadow values in a shadow cell.
const uptr kShadowCnt = 4;

// That many user bytes are mapped onto a single shadow cell.
const uptr kShadowCell = 8;

// Single shadow value.
typedef u64 RawShadow;
const uptr kShadowSize = sizeof(RawShadow);

// Shadow memory is kShadowMultiplier times larger than user memory.
const uptr kShadowMultiplier = kShadowSize * kShadowCnt / kShadowCell;

// That many user bytes are mapped onto a single meta shadow cell.
// Must be less or equal to minimal memory allocator alignment.
const uptr kMetaShadowCell = 8;

// Size of a single meta shadow value (u32).
const uptr kMetaShadowSize = 4;

// All addresses and PCs are assumed to be compressable to that many bits.
const uptr kCompressedAddrBits = 44;

#if TSAN_NO_HISTORY
const bool kCollectHistory = false;
#else
const bool kCollectHistory = true;
#endif

// The following "build consistency" machinery ensures that all source files
// are built in the same configuration. Inconsistent builds lead to
// hard to debug crashes.
#if SANITIZER_DEBUG
void build_consistency_debug();
#else
void build_consistency_release();
#endif

static inline void USED build_consistency() {
#if SANITIZER_DEBUG
  build_consistency_debug();
#else
  build_consistency_release();
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
T RoundUp(T p, u64 align) {
  DCHECK_EQ(align & (align - 1), 0);
  return (T)(((u64)p + align - 1) & ~(align - 1));
}

template<typename T>
T RoundDown(T p, u64 align) {
  DCHECK_EQ(align & (align - 1), 0);
  return (T)((u64)p & ~(align - 1));
}

// Zeroizes high part, returns 'bits' lsb bits.
template<typename T>
T GetLsb(T v, int bits) {
  return (T)((u64)v & ((1ull << bits) - 1));
}

struct MD5Hash {
  u64 hash[2];
  bool operator==(const MD5Hash &other) const;
};

MD5Hash md5_hash(const void *data, uptr size);

struct Processor;
struct ThreadState;
class ThreadContext;
struct Context;
struct ReportStack;
class ReportDesc;
class RegionAlloc;

typedef uptr AccessType;

enum : AccessType {
  kAccessWrite = 0,
  kAccessRead = 1 << 0,
  kAccessAtomic = 1 << 1,
  kAccessVptr = 1 << 2,  // read or write of an object virtual table pointer
  kAccessFree = 1 << 3,  // synthetic memory access during memory freeing
  kAccessExternalPC = 1 << 4,  // access PC can have kExternalPCBit set
};

// Descriptor of user's memory block.
struct MBlock {
  u64  siz : 48;
  u64  tag : 16;
  StackID stk;
  Tid tid;
};

COMPILER_CHECK(sizeof(MBlock) == 16);

enum ExternalTag : uptr {
  kExternalTagNone = 0,
  kExternalTagSwiftModifyingAccess = 1,
  kExternalTagFirstUserAvailable = 2,
  kExternalTagMax = 1024,
  // Don't set kExternalTagMax over 65,536, since MBlock only stores tags
  // as 16-bit values, see tsan_defs.h.
};

enum MutexType {
  MutexTypeTrace = MutexLastCommon,
  MutexTypeReport,
  MutexTypeSyncVar,
  MutexTypeAnnotations,
  MutexTypeAtExit,
  MutexTypeFired,
  MutexTypeRacy,
  MutexTypeGlobalProc,
  MutexTypeInternalAlloc,
};

}  // namespace __tsan

#endif  // TSAN_DEFS_H
