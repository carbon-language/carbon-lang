// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_
#define CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_

#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

// Detect whether we can use SIMD accelerated implementations of the control
// groups, and include the relevant platform specific APIs for the SIMD
// implementations.
//
// Reference documentation for the SIMD APIs used here:
// - https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
// - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
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

// This takes a collection of bits representing the results of looking for a
// particular tag in this metadata group and determines the first position with
// a match. The position is represented by either the least significant set bit
// or the least significant non-zero byte, depending on `ByteEncoding`. When
// represented with a non-zero byte, that byte must have at least its most
// significant bit set, but may have other bits set to any value. Bits more
// significant than the match may have any value provided there is at least one
// match. Zero matches must be represented by a zero input.
//
// Some bits of the underlying value may be known-zero, which can optimize
// various operations. These can be represented as a `ZeroMask`.
template <typename BitsInputT, bool ByteEncodingInput, BitsInputT ZeroMask = 0>
class BitIndex
    : public Printable<BitIndex<BitsInputT, ByteEncodingInput, ZeroMask>> {
 public:
  using BitsT = BitsInputT;
  static constexpr bool ByteEncoding = ByteEncodingInput;

  BitIndex() = default;
  explicit BitIndex(BitsT bits) : bits_(bits) {}

  friend auto operator==(BitIndex lhs, BitIndex rhs) -> bool {
    if (lhs.empty() || rhs.empty()) {
      return lhs.empty() == rhs.empty();
    }
    // For non-empty bit indices, compare the indices directly to ignore other
    // (extraneous) parts of the incoming bits.
    return lhs.index() == rhs.index();
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", bits_);
  }

  explicit operator bool() const { return !empty(); }

  // Returns true when there are no matches for the tag.
  auto empty() const -> bool {
    CARBON_DCHECK((bits_ & ZeroMask) == 0, "Unexpected non-zero bits!");
    __builtin_assume((bits_ & ZeroMask) == 0);
    return bits_ == 0;
  }

  // Returns the index of the first matched tag.
  auto index() -> ssize_t {
    CARBON_DCHECK(bits_ != 0, "Cannot get an index from zero bits!");
    __builtin_assume(bits_ != 0);
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
    CARBON_DCHECK(bits_ != 0, "Cannot get an index from zero bits!");
    __builtin_assume(bits_ != 0);
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
      return static_cast<size_t>(llvm::countr_zero(bits_));
    } else {
      // The index is encoded in the high bit of each byte. We compute the index
      // by counting the number of low zero bytes there are before the first
      // byte with its high bit set. Rather that shifting the high bit to be the
      // low bit and counting the trailing (least significant) zero bits
      // directly, we instead byte-reverse the bits and count the *leading*
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
      return static_cast<size_t>(llvm::countl_zero(llvm::byteswap(bits_)));
    }
  }

  BitsT bits_ = 0;
};

// This is like `BitIndex`, but allows iterating through all of the matches.
//
// A key requirement for efficient iteration is that all of the matches are
// represented with a single bit and there are no other bits set. For example,
// with byte-encoded bit indices, exactly the high bit and no other bit of each
// matching byte must be set. This is a stricter constraint than what `BitIndex`
// alone would impose on any one of the matches.
template <typename BitIndexT, BitIndexT::BitsT ByteEncodingMask = 0>
class BitIndexRange
    : public Printable<BitIndexRange<BitIndexT, ByteEncodingMask>> {
 public:
  using BitsT = BitIndexT::BitsT;
  static_assert(BitIndexT::ByteEncoding || ByteEncodingMask == 0,
                "Non-byte encoding must not have a byte encoding mask.");

  class Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          ssize_t, ssize_t> {
   public:
    Iterator() = default;
    explicit Iterator(BitsT bits) : bits_(bits) {}

    friend auto operator==(const Iterator& lhs, const Iterator& rhs) -> bool {
      return lhs.bits_ == rhs.bits_;
    }

    auto operator*() -> ssize_t& {
      CARBON_DCHECK(bits_ != 0, "Cannot get an index from zero bits!");
      __builtin_assume(bits_ != 0);
      index_ = BitIndexT(bits_).index();
      // Note that we store the index in a member so we can return a reference
      // to it here as required to be a forward iterator.
      return index_;
    }

    template <typename T>
    auto index_ptr(T* pointer) -> T* {
      return BitIndexT(bits_).index_ptr(pointer);
    }

    auto operator++() -> Iterator& {
      CARBON_DCHECK(bits_ != 0, "Must not increment past the end!");
      __builtin_assume(bits_ != 0);

      if constexpr (ByteEncodingMask != 0) {
        // Apply an increment mask to the bits first. This is used with the byte
        // encoding when the mask isn't needed until we begin incrementing.
        bits_ &= ByteEncodingMask;
      }

      // Clears the least significant set bit, effectively stepping to the next
      // match.
      bits_ &= (bits_ - 1);
      return *this;
    }

   private:
    ssize_t index_;
    BitsT bits_ = 0;
  };

  BitIndexRange() = default;
  explicit BitIndexRange(BitsT bits) : bits_(bits) {}

  explicit operator bool() const { return !empty(); }
  auto empty() const -> bool { return BitIndexT(bits_).empty(); }

  auto begin() const -> Iterator { return Iterator(bits_); }
  auto end() const -> Iterator { return Iterator(); }

  friend auto operator==(BitIndexRange lhs, BitIndexRange rhs) -> bool {
    if constexpr (ByteEncodingMask == 0) {
      // If there is no encoding mask, we can just compare the bits directly.
      return lhs.bits_ == rhs.bits_;
    } else {
      // Otherwise, compare the initial bit indices and the masked bits.
      return BitIndexT(lhs.bits_) == BitIndexT(rhs.bits_) &&
             (lhs.bits_ & ByteEncodingMask) == (rhs.bits_ & ByteEncodingMask);
    }
  }

  // Define heterogeneous equality between a masked (the current type) and
  // unmasked range. Requires a non-zero mask to avoid a redundant definition
  // with the homogeneous equality.
  friend auto operator==(BitIndexRange lhs, BitIndexRange<BitIndexT, 0> rhs)
      -> bool
    requires(ByteEncodingMask != 0)
  {
    // For mixed masked / unmasked comparison, we make sure the initial indices
    // are the same and that the masked side (LHS) is the same after masking as
    // the unmasked side (RHS).
    return BitIndexT(lhs.bits_) == BitIndexT(rhs.bits_) &&
           (lhs.bits_ & ByteEncodingMask) == rhs.bits_;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", bits_);
  }

  explicit operator BitsT() const { return bits_; }
  explicit operator BitIndexT() const { return BitIndexT(bits_); }

 private:
  template <typename FriendBitIndexT,
            FriendBitIndexT::BitsT FriendByteEncodingMask>
  friend class BitIndexRange;

  BitsT bits_ = 0;
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
//     CARBON_DCHECK(result == portable_result, "{0}", ...);
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

  // Whether to use a SIMD implementation. Even when we *support* a SIMD
  // implementation, we do not always have to use it in the event that it is
  // less efficient than the portable version.
  static constexpr bool UseSIMD =
#if CARBON_X86_SIMD_SUPPORT
      true;
#else
      false;
#endif

  // Some architectures make it much more efficient to build the match indices
  // in a byte-encoded form rather than a bit-encoded form. This encoding
  // changes verification and other aspects of our algorithms.
  static constexpr bool ByteEncoding =
#if CARBON_X86_SIMD_SUPPORT
      false;
#else
      true;
#endif
  static_assert(!ByteEncoding || Size == 8,
                "We can only support byte encoding with a group size of 8.");

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

  // Most and least significant bits set.
  static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;
  static constexpr uint64_t LSBs = 0x0101'0101'0101'0101ULL;

  using MatchIndex =
      BitIndex<std::conditional_t<ByteEncoding, uint64_t, uint32_t>,
               ByteEncoding,
               /*ZeroMask=*/ByteEncoding ? 0 : (~0U << Size)>;

  // Only one kind of portable matched range is needed.
  using PortableMatchRange = BitIndexRange<MatchIndex>;

  // We use specialized match range types for SIMD implementations to allow
  // deferring the masking operation where useful. When that optimization
  // doesn't apply, these will be the same type.
  using SIMDMatchRange =
      BitIndexRange<MatchIndex, /*ByteEncodingMask=*/ByteEncoding ? MSBs : 0>;
  using SIMDMatchPresentRange = BitIndexRange<MatchIndex>;

  // The public API range types can be either the portable or SIMD variations,
  // selected here.
  using MatchRange =
      std::conditional_t<UseSIMD, SIMDMatchRange, PortableMatchRange>;
  using MatchPresentRange =
      std::conditional_t<UseSIMD, SIMDMatchPresentRange, PortableMatchRange>;

  union {
    uint8_t metadata_bytes[Size];
    uint64_t metadata_ints[Size / 8];
#if CARBON_NEON_SIMD_SUPPORT
    uint8x8_t metadata_vec = {};
    static_assert(sizeof(metadata_vec) == Size);
#elif CARBON_X86_SIMD_SUPPORT
    __m128i metadata_vec = {};
    static_assert(sizeof(metadata_vec) == Size);
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
  // empty value.
  //
  // Note that this must only be called when `FastByteClear` is true -- in all
  // other cases users of this class should arrange to clear individual bytes in
  // the underlying array rather than using the group API. This is checked by a
  // static_assert, and the function is templated so that it is not instantiated
  // in the cases where it would not be valid.
  template <bool IsCalled = true>
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
  auto MatchPresent() const -> MatchPresentRange;

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

  using MatchBitsT = MatchIndex::BitsT;

  // A helper function to allow deducing the return type from the selected arm
  // of a `constexpr` ternary.
  template <bool Condition, typename LeftT, typename RightT>
  static auto ConstexprTernary(LeftT lhs, RightT rhs) {
    if constexpr (Condition) {
      return lhs;
    } else {
      return rhs;
    }
  }

  static auto CompareEqual(MetadataGroup lhs, MetadataGroup rhs) -> bool;

  // Functions for validating the returned matches agree with what is predicted
  // by the `byte_match` function. These either `CHECK`-fail or return true. To
  // pass validation, the `*_bits` argument must have `0x80` for those bytes
  // where `byte_match` returns true, and `0` for the rest.

  // `VerifyIndexBits` is for functions that return `MatchIndex`, as they only
  // promise to return accurate information up to the first match.
  auto VerifyIndexBits(
      MatchBitsT index_bits,
      llvm::function_ref<auto(uint8_t byte)->bool> byte_match) const -> bool;
  // `VerifyPortableRangeBits` is for functions that return `MatchRange`, and so
  // it validates all the bytes of `range_bits`.
  auto VerifyPortableRangeBits(
      MatchBitsT range_bits,
      llvm::function_ref<auto(uint8_t byte)->bool> byte_match) const -> bool;

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

  auto PortableMatch(uint8_t tag) const -> PortableMatchRange;
  auto PortableMatchPresent() const -> PortableMatchRange;

  auto PortableMatchEmpty() const -> MatchIndex;
  auto PortableMatchDeleted() const -> MatchIndex;

  static auto PortableCompareEqual(MetadataGroup lhs, MetadataGroup rhs)
      -> bool;

  // SIMD implementations of each operation. We minimize platform-specific APIs
  // to reduce the scope of errors that can only be discovered building on one
  // platform, so the bodies of these contain the platform specific code. Their
  // behavior and semantics exactly match the documentation for the un-prefixed
  // functions.
  //
  // These routines don't directly verify their results as we can build simpler
  // debug checks by comparing them against the verified portable results.
  static auto SIMDLoad(const uint8_t* metadata, ssize_t index) -> MetadataGroup;
  auto SIMDStore(uint8_t* metadata, ssize_t index) const -> void;

  auto SIMDClearDeleted() -> void;

  auto SIMDMatch(uint8_t tag) const -> SIMDMatchRange;
  auto SIMDMatchPresent() const -> SIMDMatchPresentRange;

  auto SIMDMatchEmpty() const -> MatchIndex;
  auto SIMDMatchDeleted() const -> MatchIndex;

  static auto SIMDCompareEqual(MetadataGroup lhs, MetadataGroup rhs) -> bool;

#if CARBON_X86_SIMD_SUPPORT
  // A common routine for x86 SIMD matching that can be used for matching
  // present, empty, and deleted bytes with equal efficiency.
  auto X86SIMDMatch(uint8_t match_byte) const -> SIMDMatchRange;
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
    std::memcpy(metadata + index, &metadata_bytes, Size);
  } else {
    SIMDStore(metadata, index);
  }
  CARBON_DCHECK(0 == std::memcmp(metadata + index, &metadata_bytes, Size));
}

template <bool IsCalled>
inline auto MetadataGroup::ClearByte(ssize_t byte_index) -> void {
  static_assert(!IsCalled || FastByteClear,
                "Only use byte clearing when fast!");
  static_assert(!IsCalled || Size == 8,
                "The clear implementation assumes an 8-byte group.");

  metadata_ints[0] &= ~(static_cast<uint64_t>(0xff) << (byte_index * 8));
}

inline auto MetadataGroup::ClearDeleted() -> void {
  MetadataGroup portable_g = *this;
  MetadataGroup simd_g = *this;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_g.PortableClearDeleted();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_g.SIMDClearDeleted();
    CARBON_DCHECK(
        simd_g == portable_g,
        "SIMD cleared group '{0}' doesn't match portable cleared group '{1}'",
        simd_g, portable_g);
  }
  *this = UseSIMD ? simd_g : portable_g;
}

inline auto MetadataGroup::Match(uint8_t tag) const -> MatchRange {
  // The caller should provide us with the present byte hash, and not set any
  // present bit tag on it so that this layer can manage tagging the high bit of
  // a present byte.
  CARBON_DCHECK((tag & PresentMask) == 0, "{0:x}", tag);

  PortableMatchRange portable_result;
  SIMDMatchRange simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatch(tag);
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatch(tag);
    CARBON_DCHECK(simd_result == portable_result,
                  "SIMD result '{0}' doesn't match portable result '{1}'",
                  simd_result, portable_result);
  }
  // Return whichever result we're using.
  return ConstexprTernary<UseSIMD>(simd_result, portable_result);
}

inline auto MetadataGroup::MatchPresent() const -> MatchPresentRange {
  PortableMatchRange portable_result;
  SIMDMatchPresentRange simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatchPresent();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatchPresent();
    CARBON_DCHECK(simd_result == portable_result,
                  "SIMD result '{0}' doesn't match portable result '{1}'",
                  simd_result, portable_result);
  }
  // Return whichever result we're using.
  return ConstexprTernary<UseSIMD>(simd_result, portable_result);
}

inline auto MetadataGroup::MatchEmpty() const -> MatchIndex {
  MatchIndex portable_result;
  MatchIndex simd_result;
  if constexpr (!UseSIMD || DebugSIMD) {
    portable_result = PortableMatchEmpty();
  }
  if constexpr (UseSIMD || DebugSIMD) {
    simd_result = SIMDMatchEmpty();
    CARBON_DCHECK(simd_result == portable_result,
                  "SIMD result '{0}' doesn't match portable result '{1}'",
                  simd_result, portable_result);
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
    CARBON_DCHECK(simd_result == portable_result,
                  "SIMD result '{0}' doesn't match portable result '{1}'",
                  simd_result, portable_result);
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

inline auto MetadataGroup::VerifyIndexBits(
    MatchBitsT index_bits,
    llvm::function_ref<auto(uint8_t byte)->bool> byte_match) const -> bool {
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    if constexpr (!ByteEncoding) {
      if (byte_match(metadata_bytes[byte_index])) {
        CARBON_CHECK(((index_bits >> byte_index) & 1) == 1,
                     "Bit not set at matching byte index: {0}", byte_index);
        // Only the first match is needed, so stop scanning once found.
        break;
      }

      CARBON_CHECK(((index_bits >> byte_index) & 1) == 0,
                   "Bit set at non-matching byte index: {0}", byte_index);
    } else {
      // `index_bits` is byte-encoded rather than bit encoded, so extract a
      // byte.
      uint8_t index_byte = (index_bits >> (byte_index * 8)) & 0xFF;
      if (byte_match(metadata_bytes[byte_index])) {
        CARBON_CHECK(
            (index_byte & 0x80) == 0x80,
            "Should have the high bit set for a matching byte, found: {0:x}",
            index_byte);
        // Only the first match is needed so stop scanning once found.
        break;
      }

      CARBON_CHECK(
          index_byte == 0,
          "Should have no bits set for an unmatched byte, found: {0:x}",
          index_byte);
    }
  }
  return true;
}

inline auto MetadataGroup::VerifyPortableRangeBits(
    MatchBitsT range_bits,
    llvm::function_ref<auto(uint8_t byte)->bool> byte_match) const -> bool {
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    if constexpr (!ByteEncoding) {
      if (byte_match(metadata_bytes[byte_index])) {
        CARBON_CHECK(((range_bits >> byte_index) & 1) == 1,
                     "Bit not set at matching byte index: {0}", byte_index);
      } else {
        CARBON_CHECK(((range_bits >> byte_index) & 1) == 0,
                     "Bit set at non-matching byte index: {0}", byte_index);
      }
    } else {
      // `range_bits` is byte-encoded rather than bit encoded, so extract a
      // byte.
      uint8_t range_byte = (range_bits >> (byte_index * 8)) & 0xFF;
      if (byte_match(metadata_bytes[byte_index])) {
        CARBON_CHECK(range_byte == 0x80,
                     "Should just have the high bit set for a matching byte, "
                     "found: {0:x}",
                     range_byte);
      } else {
        CARBON_CHECK(
            range_byte == 0,
            "Should have no bits set for an unmatched byte, found: {0:x}",
            range_byte);
      }
    }
  }
  return true;
}

inline auto MetadataGroup::PortableLoad(const uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup g;
  static_assert(sizeof(g) == Size);
  std::memcpy(&g.metadata_bytes, metadata + index, Size);
  return g;
}

inline auto MetadataGroup::PortableStore(uint8_t* metadata, ssize_t index) const
    -> void {
  std::memcpy(metadata + index, &metadata_bytes, Size);
}

inline auto MetadataGroup::PortableClearDeleted() -> void {
  for (uint64_t& metadata_int : metadata_ints) {
    // Deleted bytes have only the least significant bits set, so to clear them
    // we only need to clear the least significant bit. And empty bytes already
    // have a clear least significant bit, so the only least significant bits we
    // need to preserve are those of present bytes. The most significant bit of
    // every present byte is set, so we take the most significant bit of each
    // byte, shift it into the least significant bit position, and bit-or it
    // with the compliment of `LSBs`. This will have ones for every bit but the
    // least significant bits, and ones for the least significant bits of every
    // present byte.
    metadata_int &= (~LSBs | metadata_int >> 7);
  }
}

inline auto MetadataGroup::PortableMatch(uint8_t tag) const -> MatchRange {
  // The caller should provide us with the present byte hash, and not set any
  // present bit tag on it so that this layer can manage tagging the high bit of
  // a present byte.
  CARBON_DCHECK((tag & PresentMask) == 0, "{0:x}", tag);

  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a simple fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t match_bits = 0;
    uint32_t bit = 1;
    uint8_t present_byte = tag | PresentMask;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (metadata_bytes[i] == present_byte) {
        match_bits |= bit;
      }
      bit <<= 1;
    }
    return MatchRange(match_bits);
  }

  // This algorithm only works for matching *present* bytes. We leverage the
  // set high bit in the present case as part of the algorithm. The whole
  // algorithm has a critical path height of 4 operations, and does 6
  // operations total on AArch64. The operation dependency graph is:
  //
  //          group | MSBs        LSBs * match_byte + MSBs
  //                 \                /
  //                 match_bits ^ broadcast
  //                            |
  //   group & MSBs        MSBs - match_bits
  //          \                /
  //        group_MSBs & match_bits
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
  uint64_t match_bits = metadata_ints[0] | MSBs;
  // Broadcast the match byte to all bytes, and mask in the present bits in the
  // MSBs of each byte. We structure this as a multiply and an add because we
  // know that the add cannot carry, and this way it can be lowered using
  // combined multiply-add instructions if available.
  uint64_t broadcast = LSBs * tag + MSBs;
  CARBON_DCHECK(broadcast == (LSBs * tag | MSBs),
                "Unexpected carry from addition!");

  // Xor the broadcast byte pattern. This makes bytes with matches become 0, and
  // clears the high-bits of non-matches. Note that if we are looking for a tag
  // with the same value as `Empty` or `Deleted`, those bytes will be zero as
  // well.
  match_bits = match_bits ^ broadcast;
  // Subtract each byte of `match_bits` from `0x80` bytes. After this, the high
  // bit will be set only for those bytes that were zero.
  match_bits = MSBs - match_bits;
  // Zero everything but the high bits, and also zero the high bits of any bytes
  // for "not present" slots in the original group. This avoids false positives
  // for `Empty` and `Deleted` bytes in the metadata.
  match_bits &= (metadata_ints[0] & MSBs);

  // At this point, `match_bits` has the high bit set for bytes where the
  // original group byte equals `tag` plus the high bit.
  CARBON_DCHECK(VerifyPortableRangeBits(
      match_bits, [&](uint8_t byte) { return byte == (tag | PresentMask); }));
  return MatchRange(match_bits);
}

inline auto MetadataGroup::PortableMatchPresent() const -> MatchRange {
  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a simple fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t match_bits = 0;
    uint32_t bit = 1;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (metadata_bytes[i] & PresentMask) {
        match_bits |= bit;
      }
      bit <<= 1;
    }
    return MatchRange(match_bits);
  }

  // Want to keep the high bit of each byte, which indicates whether that byte
  // represents a present slot.
  uint64_t match_bits = metadata_ints[0] & MSBs;

  CARBON_DCHECK(VerifyPortableRangeBits(
      match_bits, [&](uint8_t byte) { return (byte & PresentMask) != 0; }));
  return MatchRange(match_bits);
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
      if (metadata_bytes[i] == Empty) {
        return MatchIndex(bit);
      }
      bit <<= 1;
    }
    return MatchIndex(0);
  }

  // This sets the high bit of every byte in `match_bits` unless the
  // corresponding metadata byte is 0. We take advantage of the fact that
  // the metadata bytes in are non-zero only if they are either:
  // - present: in which case the high bit of the byte will already be set; or
  // - deleted: in which case the byte will be 1, and shifting it left by 7 will
  //   cause the high bit to be set.
  uint64_t match_bits = metadata_ints[0] | (metadata_ints[0] << 7);
  // This inverts the high bits of the bytes, and clears the remaining bits.
  match_bits = ~match_bits & MSBs;

  // The high bits of the bytes of `match_bits` are set if the corresponding
  // metadata byte is `Empty`.
  CARBON_DCHECK(
      VerifyIndexBits(match_bits, [](uint8_t byte) { return byte == Empty; }));
  return MatchIndex(match_bits);
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
      if (metadata_bytes[i] == Deleted) {
        return MatchIndex(bit);
      }
      bit <<= 1;
    }
    return MatchIndex(0);
  }

  // This sets the high bit of every byte in `match_bits` unless the
  // corresponding metadata byte is 1. We take advantage of the fact that the
  // metadata bytes are not 1 only if they are either:
  // - present: in which case the high bit of the byte will already be set; or
  // - empty: in which case the byte will be 0, and in that case inverting and
  //   shifting left by 7 will have the high bit set.
  uint64_t match_bits = metadata_ints[0] | (~metadata_ints[0] << 7);
  // This inverts the high bits of the bytes, and clears the remaining bits.
  match_bits = ~match_bits & MSBs;

  // The high bits of the bytes of `match_bits` are set if the corresponding
  // metadata byte is `Deleted`.
  CARBON_DCHECK(VerifyIndexBits(match_bits,
                                [](uint8_t byte) { return byte == Deleted; }));
  return MatchIndex(match_bits);
}

inline auto MetadataGroup::PortableCompareEqual(MetadataGroup lhs,
                                                MetadataGroup rhs) -> bool {
  return llvm::equal(lhs.metadata_bytes, rhs.metadata_bytes);
}

inline auto MetadataGroup::SIMDLoad(const uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup g;
#if CARBON_NEON_SIMD_SUPPORT
  g.metadata_vec = vld1_u8(metadata + index);
#elif CARBON_X86_SIMD_SUPPORT
  g.metadata_vec =
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
  vst1_u8(metadata + index, metadata_vec);
#elif CARBON_X86_SIMD_SUPPORT
  _mm_store_si128(reinterpret_cast<__m128i*>(metadata + index), metadata_vec);
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
  metadata_ints[0] &= (~LSBs | metadata_ints[0] >> 7);
#elif CARBON_X86_SIMD_SUPPORT
  // For each byte, use `metadata_vec` if the byte's high bit is set (indicating
  // it is present), otherwise (it is empty or deleted) replace it with zero
  // (representing empty).
  metadata_vec =
      _mm_blendv_epi8(_mm_setzero_si128(), metadata_vec, metadata_vec);
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
}

inline auto MetadataGroup::SIMDMatch(uint8_t tag) const -> SIMDMatchRange {
  SIMDMatchRange result;
#if CARBON_NEON_SIMD_SUPPORT
  // Broadcast byte we want to match to every byte in the vector.
  auto match_byte_vec = vdup_n_u8(tag | PresentMask);
  // Result bytes have all bits set for the bytes that match, so we have to
  // clear everything but MSBs next.
  auto match_byte_cmp_vec = vceq_u8(metadata_vec, match_byte_vec);
  uint64_t match_bits = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
  // Note that the range will lazily mask to the MSBs as part of incrementing.
  result = SIMDMatchRange(match_bits);
#elif CARBON_X86_SIMD_SUPPORT
  result = X86SIMDMatch(tag | PresentMask);
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
  static_cast<void>(tag);
#endif
  return result;
}

inline auto MetadataGroup::SIMDMatchPresent() const -> SIMDMatchPresentRange {
  SIMDMatchPresentRange result;
#if CARBON_NEON_SIMD_SUPPORT
  // Just extract the metadata directly.
  uint64_t match_bits = vreinterpret_u64_u8(metadata_vec)[0];
  // Even though the Neon SIMD range will do its own masking, we have to mask
  // here so that `empty` is correct.
  result = SIMDMatchPresentRange(match_bits & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  // We arranged the byte vector so that present bytes have the high bit set,
  // which this instruction extracts.
  result = SIMDMatchPresentRange(_mm_movemask_epi8(metadata_vec));
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
  return result;
}

inline auto MetadataGroup::SIMDMatchEmpty() const -> MatchIndex {
  MatchIndex result;
#if CARBON_NEON_SIMD_SUPPORT
  // Compare all bytes with zero, as that is the empty byte value. Result will
  // have all bits set for any input zero byte, so we zero all but the high bits
  // below.
  auto cmp_vec = vceqz_u8(metadata_vec);
  uint64_t metadata_bits = vreinterpret_u64_u8(cmp_vec)[0];
  // The matched range is likely to be tested for zero by the caller, and that
  // test can often be folded into masking the bits with `MSBs` when we do that
  // mask in the scalar domain rather than the SIMD domain. So we do the mask
  // here rather than above prior to extracting the match bits.
  result = MatchIndex(metadata_bits & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  // Even though we only need the first match rather than all matches, we don't
  // have a more efficient way to compute this on x86 and so we reuse the
  // general match infrastructure that computes all matches in a bit-encoding.
  // We then convert it into a `MatchIndex` that just finds the first one.
  result = static_cast<MatchIndex>(X86SIMDMatch(Empty));
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
  return result;
}

inline auto MetadataGroup::SIMDMatchDeleted() const -> MatchIndex {
  MatchIndex result;
#if CARBON_NEON_SIMD_SUPPORT
  // Broadcast the `Deleted` byte across the vector and compare the bytes of
  // that with the metadata vector. The result will have all bits set for any
  // input zero byte, so we zero all but the high bits below.
  auto cmp_vec = vceq_u8(metadata_vec, vdup_n_u8(Deleted));
  uint64_t match_bits = vreinterpret_u64_u8(cmp_vec)[0];
  // The matched range is likely to be tested for zero by the caller, and that
  // test can often be folded into masking the bits with `MSBs` when we do that
  // mask in the scalar domain rather than the SIMD domain. So we do the mask
  // here rather than above prior to extracting the match bits.
  result = MatchIndex(match_bits & MSBs);
#elif CARBON_X86_SIMD_SUPPORT
  // Even though we only need the first match rather than all matches, we don't
  // have a more efficient way to compute this on x86 and so we reuse the
  // general match infrastructure that computes all matches in a bit-encoding.
  // We then convert it into a `MatchIndex` that just finds the first one.
  result = static_cast<MatchIndex>(X86SIMDMatch(Deleted));
#else
  static_assert(!UseSIMD && !DebugSIMD, "Unimplemented SIMD operation");
#endif
  return result;
}

inline auto MetadataGroup::SIMDCompareEqual(MetadataGroup lhs,
                                            MetadataGroup rhs) -> bool {
#if CARBON_NEON_SIMD_SUPPORT
  return vreinterpret_u64_u8(vceq_u8(lhs.metadata_vec, rhs.metadata_vec))[0] ==
         static_cast<uint64_t>(-1LL);
#elif CARBON_X86_SIMD_SUPPORT
  // Different x86 SIMD extensions provide different comparison functionality
  // available.
#if __SSE4_2__
  // With SSE 4.2, we can directly test and branch in the SIMD domain on whether
  // the two metadata vectors are equal.
  return _mm_testc_si128(_mm_cmpeq_epi8(lhs.metadata_vec, rhs.metadata_vec),
                         _mm_set1_epi8(0xff)) == 1;
#else
  // With older versions of SSE we have to extract the result of the comparison,
  // much like we do when matching. That will have the usual bitmask
  // representing equal bytes, and test for that exact bitmask in scalar code.
  return _mm_movemask_epi8(_mm_cmpeq_epi8(lhs.metadata_vec,
                                          rhs.metadata_vec)) == 0x0000'ffffU;
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
  // Broadcast the byte we're matching against to all bytes in a vector, and
  // compare those bytes with the metadata vector bytes.
  auto match_byte_vec = _mm_set1_epi8(match_byte);
  auto match_byte_cmp_vec = _mm_cmpeq_epi8(metadata_vec, match_byte_vec);
  // Extract the result of each byte-wise comparison into the low bits of an
  // integer.
  uint32_t match_bits = _mm_movemask_epi8(match_byte_cmp_vec);
  return MatchRange(match_bits);
}
#endif

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_
