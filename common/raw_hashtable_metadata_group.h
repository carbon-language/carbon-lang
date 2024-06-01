// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_
#define CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_

#include <cstddef>
#include <cstring>
#include <iterator>

#include "common/check.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if defined(__SSSE3__)
#include <x86intrin.h>
#define CARBON_X86_SIMD_SUPPORT 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define CARBON_NEON_SIMD_SUPPORT 1
#endif

// This namespace collects low-level utilities for implementing hashtable
// data structures. This file only provides one of them:
//
// - Primitives to manage "groups" of hashtable entries that have densely packed
//   control bytes we can scan rapidly as a group, often using SIMD facilities
//   to process the entire group at once.
namespace Carbon::RawHashtable {

// We define a constant max group size. The particular group size used in
// practice may vary, but we want to have some upper bound used to ensure
// memory allocation is done consistently across different architectures.
constexpr ssize_t MaxGroupSize = 16;

// This takes a mask representing the results of looking for a particular tag in
// this metadata group and determines the first position with a match. The
// position is represented by either the least significant set bit or the least
// significant non-zero byte, depending on `ByteEncoding`. When represented with
// a non-zero byte, that byte must have at least its most significant bit set,
// but may have other bits set to any value. Bits more significant than the
// match may have any value provided there is at least one match. Zero matches
// must be represented by a zero input.
//
// Some bits of the underlying value may be known-zero, which can optimize
// various operations. These can be represented as a `ZeroMask`.
template <typename MaskInputT, bool ByteEncoding, MaskInputT ZeroMask = 0>
class BitIndex
    : public Printable<BitIndex<MaskInputT, ByteEncoding, ZeroMask>> {
 public:
  using MaskT = MaskInputT;

  BitIndex() = default;
  explicit BitIndex(MaskT mask) : mask_(mask) {}

  friend auto operator==(BitIndex lhs, BitIndex rhs) -> bool {
    if (lhs.empty() || rhs.empty()) {
      return lhs.empty() == rhs.empty();
    }
    // For non-empty bit indices, compare the indices directly to ignore other
    // (extraneous) parts of the mask.
    return lhs.index() == rhs.index();
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", mask_);
  }

  explicit operator bool() const { return !empty(); }

  // Returns true when there are no matches for the tag.
  auto empty() const -> bool {
    CARBON_DCHECK((mask_ & ZeroMask) == 0) << "Unexpected non-zero bits!";
    __builtin_assume((mask_ & ZeroMask) == 0);
    return mask_ == 0;
  }

  // Returns the index of the first matched tag.
  auto index() -> ssize_t {
    CARBON_DCHECK(mask_ != 0) << "Cannot get an index from a zero mask!";
    __builtin_assume(mask_ != 0);
    ssize_t index = unscaled_index();

    if constexpr (ByteEncoding) {
      // Shift to scale out of the byte encoding.
      index >>= ByteEncodingShift;
    }

    return index;
  }

  // Optimized tool to index a pointer `p` by `index()`.
  template <typename T>
  auto index_ptr(T* pointer) -> T* {
    CARBON_DCHECK(mask_ != 0) << "Cannot get an index from a zero mask!";
    __builtin_assume(mask_ != 0);
    if constexpr (!ByteEncoding) {
      return &pointer[unscaled_index()];
    }

    ssize_t index = unscaled_index();

    // Scale the index as we counted zero *bits* and not zero *bytes*.
    // However, we can fold that scale with the size of `T` when it is a power
    // of two or divisible by 8.
    CARBON_DCHECK(
        (index & ((static_cast<size_t>(1) << ByteEncodingShift) - 1)) == 0);
    if constexpr (sizeof(T) % 8 == 0) {
      constexpr size_t FoldedScale = sizeof(T) / 8;
      index *= FoldedScale;
      return reinterpret_cast<T*>(
          &reinterpret_cast<std::byte*>(pointer)[index]);
    } else if constexpr (llvm::isPowerOf2_64(sizeof(T))) {
      constexpr size_t ScaleShift = llvm::CTLog2<sizeof(T)>();
      static_assert(ScaleShift <= ByteEncodingShift,
                    "Scaling by >=8 should be handled above!");
      constexpr size_t FoldedShift = ByteEncodingShift - ScaleShift;
      index >>= FoldedShift;
      return reinterpret_cast<T*>(
          &reinterpret_cast<std::byte*>(pointer)[index]);
    }

    // Nothing we can fold here.
    return &pointer[index >> ByteEncodingShift];
  }

 private:
  // When using a byte encoding, we'll need to shift any index by this amount.
  static constexpr size_t ByteEncodingShift = 3;

  auto unscaled_index() -> ssize_t {
    if constexpr (!ByteEncoding) {
      // Note the cast to `size_t` to force zero extending the result.
      return static_cast<size_t>(llvm::countr_zero(mask_));
    } else {
      // The index is encoded in the high bit of each byte. We compute the index
      // by counting the number of low zero bytes there are before the first
      // byte with its high bit set. Rather that shifting the high bit to be the
      // low bit and counting the trailing (least significant) zero bits
      // directly, we instead byte-reverse the mask and count the *leading*
      // (most significant) zero bits. While this may be a wash on CPUs with
      // direct support for counting the trailing zero bits, AArch64 only
      // supports counting the leading zero bits and requires a bit-reverse to
      // count the trailing zero bits. Doing the byte-reverse approach
      // essentially combines moving the high bit into the low bit and the
      // reverse necessary for counting the zero bits. While this only removes
      // one instruction, it is an instruction in the critical path of the
      // hottest part of table lookup, and that critical path dependency height
      // is few enough instructions that removing even one significantly impacts
      // latency.
      //
      // We also cast to `size_t` to clearly zero-extend the result.
      return static_cast<size_t>(llvm::countl_zero(llvm::byteswap(mask_)));
    }
  }

  MaskT mask_ = 0;
};

// This is like `BitIndex`, but allows iterating through all of the matches.
//
// A key requirement for efficient iteration is that all of the matches are
// represented with a single bit and there are no other bits set. For example,
// with byte-encoded bit indices, exactly the high bit and no other bit of each
// matching byte must be set. This is a stricter constraint than what `BitIndex`
// alone would impose on any one of the matches.
template <typename BitIndexT>
class BitIndexRange : public Printable<BitIndexRange<BitIndexT>> {
 public:
  using MaskT = BitIndexT::MaskT;

  class Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          ssize_t, ssize_t> {
   public:
    Iterator() = default;
    explicit Iterator(MaskT mask) : mask_(mask) {}

    auto operator==(const Iterator& rhs) const -> bool {
      return mask_ == rhs.mask_;
    }

    auto operator*() -> ssize_t& {
      CARBON_DCHECK(mask_ != 0) << "Cannot get an index from a zero mask!";
      __builtin_assume(mask_ != 0);
      index_ = BitIndexT(mask_).index();
      return index_;
    }

    template <typename T>
    auto index_ptr(T* pointer) -> T* {
      return BitIndexT(mask_).index_ptr(pointer);
    }

    auto operator++() -> Iterator& {
      CARBON_DCHECK(mask_ != 0) << "Must not increment past the end!";
      __builtin_assume(mask_ != 0);
      // Clears the least significant set bit, effectively stepping to the next
      // match.
      mask_ &= (mask_ - 1);
      return *this;
    }

   private:
    ssize_t index_;
    MaskT mask_ = 0;
  };

  BitIndexRange() = default;
  explicit BitIndexRange(MaskT mask) : mask_(mask) {}

  explicit operator bool() const { return !empty(); }
  auto empty() const -> bool { return BitIndexT(mask_).empty(); }

  auto begin() const -> Iterator { return Iterator(mask_); }
  auto end() const -> Iterator { return Iterator(); }

  friend auto operator==(BitIndexRange lhs, BitIndexRange rhs) -> bool {
    return lhs.mask_ == rhs.mask_;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", mask_);
  }

  explicit operator MaskT() const { return mask_; }
  explicit operator BitIndexT() const { return BitIndexT(mask_); }

 private:
  MaskT mask_ = 0;
};

// A group of metadata bytes that can be manipulated together.
//
// The metadata bytes used Carbon's hashtable implementation are designed to
// support being manipulating as groups, either using architecture specific SIMD
// code sequences or using portable SIMD-in-an-integer-register code sequences.
// These operations are unusually performance sensitive and in sometimes
// surprising ways. The implementations here are crafted specifically to
// optimize the particular usages in Carbon's hashtable and should not be
// expected to be reusable in any other context.
//
// Throughout the functions operating on this type we try to use patterns with a
// fallback portable implementation which can be directly used in the absence of
// a SIMD implementation, but is also used (with the same code) to check that
// any SIMD implementation produces the same result as the portable one. These
// patterns help minimize un-compiled or un-tested paths through either portable
// or SIMD code, regardless of which path is actually *used* on a particular
// platform. To illustrate a common version of this pattern, we might have code
// like:
//
// ```cpp
// auto MetadataGroup::Operation(...) -> ... {
//   ... portable_result;
//   ... simd_result;
//   if constexpr (!UseSIMD || DebugSIMD) {
//     portable_result = PortableOperation(...);
//   }
//   if (UseSIMD || DebugSIMD) {
//     simd_result = SIMDOperation(...)
//     CARBON_DCHECK(result == portable_result) << ...;
//   }
//   return UseSIMD ? simd_result : portable_result;
// }
// ```
class MetadataGroup : public Printable<MetadataGroup> {
 public:
  static constexpr ssize_t Size =
#if CARBON_X86_SIMD_SUPPORT
      16;
#else
      8;
#endif
  static_assert(Size >= 8);
  static_assert(Size % 8 == 0);
  static_assert(Size <= MaxGroupSize);
  static_assert(MaxGroupSize % Size == 0);
  static_assert(llvm::isPowerOf2_64(Size),
                "The group size must be a constant power of two so dividing by "
                "it is a simple shift.");
  static constexpr ssize_t Mask = Size - 1;

  // Each control byte can have special values. All special values have the
  // most significant bit cleared to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  static constexpr uint8_t PresentMask = 0b1000'0000;

  // We need to indicate to users of the metadata group when they can hold a
  // group value in a "register" (local variable) across clearing of individual
  // bytes in the group efficiently. If the entire group can fit in an integer
  // register, this works well and clients of the group should work to use the
  // already-loaded value when clearing bytes. But when we have a larger group
  // size, clearing the byte will typically require storing a byte to memory and
  // re-loading the group. The usage patterns that need to clear bytes can in
  // those cases avoid clearing a loaded group, and clear the byte directly in
  // the larger metadata array.
  static constexpr bool FastByteClear = Size == 8;

  using MatchIndex =
#if CARBON_X86_SIMD_SUPPORT
      BitIndex<uint32_t, /*ByteEncoding=*/false, /*ZeroMask=*/0xFFFF0000>;
#else
      BitIndex<uint64_t, /*ByteEncoding=*/true>;
#endif

  using MatchRange = BitIndexRange<MatchIndex>;

  union {
    uint8_t bytes[Size];
    uint64_t byte_ints[Size / 8];
#if CARBON_NEON_SIMD_SUPPORT
    uint8x8_t byte_vec = {};
    static_assert(sizeof(byte_vec) == Size);
#elif CARBON_X86_SIMD_SUPPORT
    __m128i byte_vec = {};
    static_assert(sizeof(byte_vec) == Size);
#endif
  };

  auto Print(llvm::raw_ostream& out) const -> void;

  friend auto operator==(MetadataGroup lhs, MetadataGroup rhs) -> bool {
    return CompareEqual(lhs, rhs);
  }

  // The main API for this class. This API will switch between a portable and
  // SIMD implementation based on what is most efficient, but in debug builds
  // will cross check that the implementations do not diverge.

  // Load and return a group of metadata bytes out of the main metadata array at
  // a particular `index`. The index must be a multiple of `GroupSize`. This
  // will arrange for the load to place the group into the correct structure for
  // efficient register-based processing.
  static auto Load(const uint8_t* metadata, ssize_t index) -> MetadataGroup;

  // Store this metadata group into the main metadata array at the provided
  // `index`. The index must be a multiple of `GroupSize`.
  auto Store(uint8_t* metadata, ssize_t index) const -> void;

  // Clear a byte of this group's metadata at the provided `byte_index` to the
  // empty value. Note that this must only be called when `FastByteClear` is
  // true -- in all other cases users of this class should arrange to clear
  // individual bytes in the underlying array rather than using the group API.
  auto ClearByte(ssize_t byte_index) -> void;

  // Clear all of this group's metadata bytes that indicate a deleted slot to
  // the empty value.
  auto ClearDeleted() -> void;

  // Find all of the bytes of metadata in this group that are present and whose
  // low 7 bits match the provided `tag`. The `tag` byte must have a clear high
  // bit, only 7 bits of tag are used. Note that this means the provided tag is
  // *not* the actual present metadata byte -- this function is responsible for
  // mapping the tag into that form as it can do so more efficiently in some
  // cases. A range over all of the byte indices which matched is returned.
  auto Match(uint8_t tag) const -> MatchRange;

  // Find all of the present bytes of metadata in this group. A range over all
  // of the byte indices which are present is returned.
  auto MatchPresent() const -> MatchRange;

  // Find the first byte of the metadata group that is empty and return that
  // index. There is no order or position required for which of the bytes of
  // metadata is considered "first", any model will do that makes it efficient
  // to produce the matching index. Must return an empty match index if no bytes
  // match the empty metadata.
  auto MatchEmpty() const -> MatchIndex;

  // Find the first byte of the metadata group that is deleted and return that
  // index. There is no order or position required for which of the bytes of
  // metadata is considered "first", any model will do that makes it efficient
  // to produce the matching index. Must return an empty match index if no bytes
  // match the deleted metadata.
  auto MatchDeleted() const -> MatchIndex;

 private:
  // Two classes only defined in the benchmark code are allowed to directly call
  // the portable and SIMD implementations for benchmarking purposes.
  friend class BenchmarkPortableMetadataGroup;
  friend class BenchmarkSIMDMetadataGroup;

  // Whether to use a SIMD implementation. Even when we *support* a SIMD
  // implementation, we do not always have to use it in the event that it is
  // less efficient than the portable version.
  static constexpr bool UseSIMD =
#if CARBON_X86_SIMD_SUPPORT
      true;
#else
      false;
#endif

  // All SIMD variants that we have an implementation for should be enabled for
  // debugging. This lets us maintain a SIMD implementation even if it is not
  // used due to performance reasons, and easily re-enable it if the performance
  // changes.
  static constexpr bool DebugSIMD =
#if !defined(NDEBUG) && (CARBON_NEON_SIMD_SUPPORT || CARBON_X86_SIMD_SUPPORT)
      true;
#else
      false;
#endif

  // Most and least significant bits set.
  static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;
  static constexpr uint64_t LSBs = 0x0101'0101'0101'0101ULL;

  using MatchMask = MatchIndex::MaskT;

  static auto CompareEqual(MetadataGroup lhs, MetadataGroup rhs) -> bool;

  // Functions for validating the returned matches agree with what is predicted
  // by the `byte_match` function. These either `CHECK`-fail or return true. To
  // pass validation, the `*_mask` argument must have `0x80` for those bytes
  // where `byte_match` returns true, and `0` for the rest.

  // `VerifyIndexMask` is for functions that return `MatchIndex`, as they only
  // promise to return accurate information up to the first match.
  auto VerifyIndexMask(
      MatchMask index_mask,
      llvm::function_ref<auto(uint8_t mask_byte)->bool> byte_match) const
      -> bool;
  // `VerifyRangeMask` is for functions that return `MatchRange`, and so it
  // validates all the bytes of `range_mask`.
  auto VerifyRangeMask(
      MatchMask range_mask,
      llvm::function_ref<auto(uint8_t mask_byte)->bool> byte_match) const
      -> bool;

  // Portable implementations of each operation. These are used on platforms
  // without SIMD support or where the portable implementation is faster than
  // SIMD. They are heavily optimized even though they are not SIMD because we
  // expect there to be platforms where the portable implementation can
  // outperform SIMD. Their behavior and semantics exactly match the
  // documentation for the un-prefixed functions.
  //
  // In debug builds, these also directly verify their results to help establish
  // baseline functionality.
  static auto PortableLoad(const uint8_t* metadata, ssize_t index)
      -> MetadataGroup;
  auto PortableStore(uint8_t* metadata, ssize_t index) const -> void;

  auto PortableClearDeleted() -> void;

  auto PortableMatch(uint8_t tag) const -> MatchRange;
  auto PortableMatchPresent() const -> MatchRange;

  auto PortableMatchEmpty() const -> MatchIndex;
  auto PortableMatchDeleted() const -> MatchIndex;

  static auto PortableCompareEqual(MetadataGroup lhs, MetadataGroup rhs)
      -> bool;

  // SIMD implementations of each operation. We minimize platform-specific APIs
  // to reduce the scope of errors that can only be discoverd building on one
  // platform, so the bodies of these contain the platform specific code. Their
  // behavior and semantics exactly match the documentation for the un-prefixed
  // functions.
  //
  // These routines don't directly verify their results as we can build simpler
  // debug checks by comparing them against the verified portable results.
  static auto SIMDLoad(const uint8_t* metadata, ssize_t index) -> MetadataGroup;
  auto SIMDStore(uint8_t* metadata, ssize_t index) const -> void;

  auto SIMDClearDeleted() -> void;

  auto SIMDMatch(uint8_t tag) const -> MatchRange;
  auto SIMDMatchPresent() const -> MatchRange;

  auto SIMDMatchEmpty() const -> MatchIndex;
  auto SIMDMatchDeleted() const -> MatchIndex;

  static auto SIMDCompareEqual(MetadataGroup lhs, MetadataGroup rhs) -> bool;

#if CARBON_X86_SIMD_SUPPORT
  // A common routine for x86 SIMD matching that can be used for matching
  // present, empty, and deleted bytes with equal efficiency.
  auto X86SIMDMatch(uint8_t match_byte) const -> MatchRange;
#endif
};

// Promote the size and mask to top-level constants as we'll need to operate on
// the grouped structure outside of the metadata bytes.
inline constexpr ssize_t GroupSize = MetadataGroup::Size;
inline constexpr ssize_t GroupMask = MetadataGroup::Mask;

inline auto MetadataGroup::Load(const uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup portable_g;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_g = PortableLoad(metadata, index);
    if constexpr (!UseSIMD) {
      return portable_g;
    }
  }
  MetadataGroup g = SIMDLoad(metadata, index);
  CARBON_DCHECK(g == portable_g);
  return g;
}

inline auto MetadataGroup::Store(uint8_t* metadata, ssize_t index) const
    -> void {
  if constexpr (!UseSIMD) {
    std::memcpy(metadata + index, &bytes, Size);
  } else {
    SIMDStore(metadata, index);
  }
  CARBON_DCHECK(0 == std::memcmp(metadata + index, &bytes, Size));
}

inline auto MetadataGroup::ClearByte(ssize_t byte_index) -> void {
  CARBON_DCHECK(FastByteClear) << "Only use byte clearing when fast!";
  CARBON_DCHECK(Size == 8)
      << "The clear implementation assumes an 8-byte group.";

  byte_ints[0] &= ~(static_cast<uint64_t>(0xff) << (byte_index * 8));
}

inline auto MetadataGroup::ClearDeleted() -> void {
  MetadataGroup portable_g = *this;
  MetadataGroup simd_g = *this;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_g.PortableClearDeleted();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_g.SIMDClearDeleted();
    CARBON_DCHECK(simd_g == portable_g)
        << "SIMD cleared group '" << simd_g
        << "' doesn't match portable cleared group '" << portable_g << "'";
  }
  *this = UseSIMD ? simd_g : portable_g;
}

inline auto MetadataGroup::Match(uint8_t tag) const -> MatchRange {
  // The caller should provide us with the present byte hash, and not set any
  // present bit tag on it so that this layer can manage tagging the high bit of
  // a present byte.
  CARBON_DCHECK((tag & PresentMask) == 0) << llvm::formatv("{0:x}", tag);

  MatchRange portable_result;
  MatchRange simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatch(tag);
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatch(tag);
    CARBON_DCHECK(simd_result == portable_result)
        << "SIMD result '" << simd_result << "' doesn't match portable result '"
        << portable_result << "'";
  }
  return UseSIMD ? simd_result : portable_result;
}

inline auto MetadataGroup::MatchPresent() const -> MatchRange {
  MatchRange portable_result;
  MatchRange simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatchPresent();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatchPresent();
    CARBON_DCHECK(simd_result == portable_result)
        << "SIMD result '" << simd_result << "' doesn't match portable result '"
        << portable_result << "'";
  }
  return UseSIMD ? simd_result : portable_result;
}

inline auto MetadataGroup::MatchEmpty() const -> MatchIndex {
  MatchIndex portable_result;
  MatchIndex simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatchEmpty();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatchEmpty();
    CARBON_DCHECK(simd_result == portable_result)
        << "SIMD result '" << simd_result << "' doesn't match portable result '"
        << portable_result << "'";
  }
  return UseSIMD ? simd_result : portable_result;
}

inline auto MetadataGroup::MatchDeleted() const -> MatchIndex {
  MatchIndex portable_result;
  MatchIndex simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatchDeleted();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatchDeleted();
    CARBON_DCHECK(simd_result == portable_result)
        << "SIMD result '" << simd_result << "' doesn't match portable result '"
        << portable_result << "'";
  }
  return UseSIMD ? simd_result : portable_result;
}

inline auto MetadataGroup::CompareEqual(MetadataGroup lhs, MetadataGroup rhs)
    -> bool {
  bool portable_result;
  bool simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableCompareEqual(lhs, rhs);
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDCompareEqual(lhs, rhs);
    CARBON_DCHECK(simd_result == portable_result);
  }
  return UseSIMD ? simd_result : portable_result;
}

inline auto MetadataGroup::VerifyIndexMask(
    MatchMask index_mask,
    llvm::function_ref<auto(uint8_t byte)->bool> byte_match) const -> bool {
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    if constexpr (Size > 8) {
      if (byte_match(bytes[byte_index])) {
        CARBON_CHECK(((index_mask >> byte_index) & 1) == 1)
            << "Bit not set at matching byte index: " << byte_index;
        // Only the first match is needed, so stop scanning once found.
        break;
      }

      CARBON_CHECK(((index_mask >> byte_index) & 1) == 0)
          << "Bit set at non-matching byte index: " << byte_index;
    } else {
      // The mask is byte-encoded rather than bit encoded, so extract a byte.
      uint8_t mask_byte = (index_mask >> (byte_index * 8)) & 0xFF;
      if (byte_match(bytes[byte_index])) {
        CARBON_CHECK((mask_byte & 0x80) == 0x80)
            << "Should have the high bit set for a matching byte, found: "
            << llvm::formatv("{0:x}", mask_byte);
        // Only the first match is needed so stop scanning once found.
        break;
      }

      CARBON_CHECK(mask_byte == 0)
          << "Should have no bits set for an unmatched byte, found: "
          << llvm::formatv("{0:x}", mask_byte);
    }
  }
  return true;
}

inline auto MetadataGroup::VerifyRangeMask(
    MatchMask range_mask,
    llvm::function_ref<auto(uint8_t byte)->bool> byte_match) const -> bool {
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    if constexpr (Size > 8) {
      if (byte_match(bytes[byte_index])) {
        CARBON_CHECK(((range_mask >> byte_index) & 1) == 1)
            << "Bit not set at matching byte index: " << byte_index;
      } else {
        CARBON_CHECK(((range_mask >> byte_index) & 1) == 0)
            << "Bit set at non-matching byte index: " << byte_index;
      }
    } else {
      // The mask is byte-encoded rather than bit encoded, so extract a byte.
      uint8_t mask_byte = (range_mask >> (byte_index * 8)) & 0xFF;
      if (byte_match(bytes[byte_index])) {
        CARBON_CHECK(mask_byte == 0x80)
            << "Should just have the high bit set for a matching byte, found: "
            << llvm::formatv("{0:x}", mask_byte);
      } else {
        CARBON_CHECK(mask_byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", mask_byte);
      }
    }
  }
  return true;
}

inline auto MetadataGroup::PortableLoad(const uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup g;
  static_assert(sizeof(g) == Size);
  std::memcpy(&g.bytes, metadata + index, Size);
  return g;
}

inline auto MetadataGroup::PortableStore(uint8_t* metadata, ssize_t index) const
    -> void {
  std::memcpy(metadata + index, &bytes, Size);
}

inline auto MetadataGroup::PortableClearDeleted() -> void {
  for (uint64_t& byte_int : byte_ints) {
    // Deleted bytes have only the least significant bits set, so to clear them
    // we only need to clear the least significant bit. And empty bytes already
    // have a clear least significant bit, so the only least significant bits we
    // need to preserve are those of present bytes. The most significant bit of
    // every present byte is set, so we take the most significant bit of each
    // byte, shift it into the least significant bit position, and bit-or it
    // with the inverse of the least-significant-bits mask. This will have ones
    // for every bit but the least significant bits, and ones for the least
    // significant bits of every present byte.
    byte_int &= (~LSBs | byte_int >> 7);
  }
}

inline auto MetadataGroup::PortableMatch(uint8_t tag) const -> MatchRange {
  // The caller should provide us with the present byte hash, and not set any
  // present bit tag on it so that this layer can manage tagging the high bit of
  // a present byte.
  CARBON_DCHECK((tag & PresentMask) == 0) << llvm::formatv("{0:x}", tag);

  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a simple fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t mask = 0;
    uint32_t bit = 1;
    uint8_t present_byte = tag | PresentMask;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (bytes[i] == present_byte) {
        mask |= bit;
      }
      bit <<= 1;
    }
    return MatchRange(mask);
  }

  // This algorithm only works for matching *present* bytes. We leverage the
  // set high bit in the present case as part of the algorithm. The whole
  // algorithm has a critical path height of 4 operations, and does 6
  // operations total on AArch64. The operation dependency graph is:
  //
  //          group | MSBs    LSBs * match_byte + MSBs
  //                 \            /
  //                 mask ^ broadcast
  //                      |
  // group & MSBs    MSBs - mask
  //        \            /
  //    group_MSBs & mask
  //
  // This diagram and the operation count are specific to AArch64 where we have
  // a fused *integer* multiply-add operation.
  //
  // While it is superficially similar to the "find zero bytes in a word" bit
  // math trick, it is different because this is designed to have no false
  // positives and perfectly produce 0x80 for matching bytes and 0x00 for
  // non-matching bytes. This is do-able because we constrain to only handle
  // present matches which only require testing 7 bits and have a particular
  // layout.

  // Set the high bit of every byte to `1`. Any matching byte is a present byte
  // and so always has this bit set as well, which means the xor below, in
  // addition to zeroing the low 7 bits of any byte that matches the tag, also
  // clears the high bit of every byte.
  uint64_t mask = byte_ints[0] | MSBs;
  // Broadcast the match byte to all bytes, and mask in the present bits in the
  // MSBs of each byte. We structure this as a multiply and an add because we
  // know that the add cannot carry, and this way it can be lowered using
  // combined multiply-add instructions if available.
  uint64_t broadcast = LSBs * tag + MSBs;
  CARBON_DCHECK(broadcast == (LSBs * tag | MSBs))
      << "Unexpected carry from addition!";

  // Xor the broadcast byte pattern. This makes bytes with matches become 0, and
  // clears the high-bits of non-matches. Note that if we are looking for a tag
  // with the same value as `Empty` or `Deleted`, those bytes will be zero as
  // well.
  mask = mask ^ broadcast;
  // Subtract the mask bytes from `0x80` bytes. After this, the high bit will be
  // set only for those bytes that were zero.
  mask = MSBs - mask;
  // Zero everything but the high bits, and also zero the high bits of any bytes
  // for "not present" slots in the original group. This avoids false positives
  // for `Empty` and `Deleted` bytes in the metadata.
  mask &= (byte_ints[0] & MSBs);

  // At this point, `mask` has the high bit set for bytes where the original
  // group byte equals `tag` plus the high bit.
  CARBON_DCHECK(VerifyRangeMask(
      mask, [&](uint8_t byte) { return byte == (tag | PresentMask); }));
  return MatchRange(mask);
}

inline auto MetadataGroup::PortableMatchPresent() const -> MatchRange {
  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a simple fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t mask = 0;
    uint32_t bit = 1;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (bytes[i] & PresentMask) {
        mask |= bit;
      }
      bit <<= 1;
    }
    return MatchRange(mask);
  }

  // Want to keep the high bit of each byte, which indicates whether that byte
  // represents a present slot.
  uint64_t mask = byte_ints[0] & MSBs;

  CARBON_DCHECK(VerifyRangeMask(
      mask, [&](uint8_t byte) { return (byte & PresentMask) != 0; }));
  return MatchRange(mask);
}

inline auto MetadataGroup::PortableMatchEmpty() const -> MatchIndex {
  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a simple fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t bit = 1;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (bytes[i] == Empty) {
        return MatchIndex(bit);
      }
      bit <<= 1;
    }
    return MatchIndex(0);
  }

  // This sets the high bit of every byte in our match mask unless the
  // corresponding metadata byte is 0. We take advantage of the fact that
  // the metadata bytes in are non-zero only if they are either:
  // - present: in which case the high bit of the byte will already be set; or
  // - deleted: in which case the byte will be 1, and shifting it left by 7 will
  //   cause the high bit to be set.
  uint64_t mask = byte_ints[0] | (byte_ints[0] << 7);
  // This inverts the high bits of the bytes, and clears the remaining bits.
  mask = ~mask & MSBs;

  // The high bits of the bytes of `mask` are set if the corresponding metadata
  // byte is `Empty`.
  CARBON_DCHECK(
      VerifyIndexMask(mask, [](uint8_t byte) { return byte == Empty; }));
  return MatchIndex(mask);
}

inline auto MetadataGroup::PortableMatchDeleted() const -> MatchIndex {
  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a simple fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t bit = 1;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (bytes[i] == Deleted) {
        return MatchIndex(bit);
      }
      bit <<= 1;
    }
    return MatchIndex(0);
  }

  // This sets the high bit of every byte in our match mask unless the
  // corresponding metadata byte is 1. We take advantage of the fact that the
  // metadata bytes are not 1 only if they are either:
  // - present: in which case the high bit of the byte will already be set; or
  // - empty: in which case the byte will be 0, and in that case inverting and
  //   shifting left by 7 will have the high bit set.
  uint64_t mask = byte_ints[0] | (~byte_ints[0] << 7);
  // This inverts the high bits of the bytes, and clears the remaining bits.
  mask = ~mask & MSBs;

  // The high bits of the bytes of `mask` are set if the corresponding metadata
  // byte is `Deleted`.
  CARBON_DCHECK(
      VerifyIndexMask(mask, [](uint8_t byte) { return byte == Deleted; }));
  return MatchIndex(mask);
}

inline auto MetadataGroup::PortableCompareEqual(MetadataGroup lhs,
                                                MetadataGroup rhs) -> bool {
  return llvm::equal(lhs.bytes, rhs.bytes);
}

inline auto MetadataGroup::SIMDLoad(const uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup g;
#if CARBON_NEON_SIMD_SUPPORT
  g.byte_vec = vld1_u8(metadata + index);
#elif CARBON_X86_SIMD_SUPPORT
  g.byte_vec =
      _mm_load_si128(reinterpret_cast<const __m128i*>(metadata + index));
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
  static_cast<void>(metadata);
  static_cast<void>(index);
#endif
  return g;
}

inline auto MetadataGroup::SIMDStore(uint8_t* metadata, ssize_t index) const
    -> void {
#if CARBON_NEON_SIMD_SUPPORT
  vst1_u8(metadata + index, byte_vec);
#elif CARBON_X86_SIMD_SUPPORT
  _mm_store_si128(reinterpret_cast<__m128i*>(metadata + index), byte_vec);
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
  static_cast<void>(metadata);
  static_cast<void>(index);
#endif
}

inline auto MetadataGroup::SIMDClearDeleted() -> void {
#if CARBON_NEON_SIMD_SUPPORT
  // There is no good Neon operation to implement this, so do it using integer
  // code. This is reasonably fast, but unfortunate because it forces the group
  // out of a SIMD register and into a general purpose register, which can have
  // high latency.
  byte_ints[0] &= (~LSBs | byte_ints[0] >> 7);
#elif CARBON_X86_SIMD_SUPPORT
  byte_vec = _mm_blendv_epi8(_mm_setzero_si128(), byte_vec, byte_vec);
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
}

inline auto MetadataGroup::SIMDMatch(uint8_t tag) const -> MatchRange {
  MatchRange result;
#if CARBON_NEON_SIMD_SUPPORT
  auto match_byte_vec = vdup_n_u8(tag | PresentMask);
  auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
  uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
  // The mask in the range is likely to be tested for zero right after this
  // routine, and that test can often be folded into an integer constant mask,
  // so rather than doing the mask in SIMD before we extract the mask, we do it
  // here to allow that combination.
  result = MatchRange(mask & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  result = X86SIMDMatch(tag | PresentMask);
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
  static_cast<void>(tag);
#endif
  return result;
}

inline auto MetadataGroup::SIMDMatchPresent() const -> MatchRange {
  MatchRange result;
#if CARBON_NEON_SIMD_SUPPORT
  // Just extract directly.
  uint64_t mask = vreinterpret_u64_u8(byte_vec)[0];
  // The mask in the range is likely to be tested for zero right after this
  // routine, and that test can often be folded into an integer constant mask,
  // so rather than doing the mask in SIMD before we extract the mask, we do it
  // here to allow that combination.
  result = MatchRange(mask & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  // We arrange the byte vector for present bytes so that we can directly
  // extract it as a mask.
  result = MatchRange(_mm_movemask_epi8(byte_vec));
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
  return result;
}

inline auto MetadataGroup::SIMDMatchEmpty() const -> MatchIndex {
  MatchIndex result;
#if CARBON_NEON_SIMD_SUPPORT
  auto cmp_vec = vceqz_u8(byte_vec);
  uint64_t mask = vreinterpret_u64_u8(cmp_vec)[0];
  // The mask in the range is likely to be tested for zero right after this
  // routine, and that test can often be folded into an integer constant mask,
  // so rather than doing the mask in SIMD before we extract the mask, we do it
  // here to allow that combination.
  result = MatchIndex(mask & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  result = static_cast<MatchIndex>(X86SIMDMatch(Empty));
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
  return result;
}

inline auto MetadataGroup::SIMDMatchDeleted() const -> MatchIndex {
  MatchIndex result;
#if CARBON_NEON_SIMD_SUPPORT
  auto cmp_vec = vceq_u8(byte_vec, vdup_n_u8(Deleted));
  uint64_t mask = vreinterpret_u64_u8(cmp_vec)[0];
  // The mask in the range is likely to be tested for zero right after this
  // routine, and that test can often be folded into an integer constant mask,
  // so rather than doing the mask in SIMD before we extract the mask, we do it
  // here to allow that combination.
  result = MatchIndex(mask & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  result = static_cast<MatchIndex>(X86SIMDMatch(Deleted));
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
  return result;
}

inline auto MetadataGroup::SIMDCompareEqual(MetadataGroup lhs,
                                            MetadataGroup rhs) -> bool {
#if CARBON_NEON_SIMD_SUPPORT
  return vreinterpret_u64_u8(vceq_u8(lhs.byte_vec, rhs.byte_vec))[0] ==
         static_cast<uint64_t>(-1LL);
#elif CARBON_X86_SIMD_SUPPORT
  // Different x86 SIMD extensions provide different comparison functionality.
#if __SSE4_2__
  return _mm_testc_si128(_mm_cmpeq_epi8(lhs.byte_vec, rhs.byte_vec),
                         _mm_set1_epi8(0xff)) == 1;
#else
  return _mm_movemask_epi8(_mm_cmpeq_epi8(lhs.byte_vec, rhs.byte_vec)) ==
         0x0000'ffffU;
#endif
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
  static_cast<void>(lhs);
  static_cast<void>(rhs);
  return false;
#endif
}

#if CARBON_X86_SIMD_SUPPORT
inline auto MetadataGroup::X86SIMDMatch(uint8_t match_byte) const
    -> MatchRange {
  auto match_byte_vec = _mm_set1_epi8(match_byte);
  auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
  uint32_t mask = _mm_movemask_epi8(match_byte_cmp_vec);
  return MatchRange(mask);
}
#endif

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_
