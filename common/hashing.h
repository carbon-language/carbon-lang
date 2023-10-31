// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_HASHING_H_
#define CARBON_COMMON_HASHING_H_

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

// A type wrapping a 64-bit hash code and provides a basic but limited API.
class HashCode : public Printable<HashCode> {
 public:
  HashCode() = default;

  constexpr explicit HashCode(uint64_t value) : value_(value) {}

  // Conversion to an actual integer is explicit to avoid accidental, unintended
  // arithmetic.
  constexpr explicit operator uint64_t() const { return value_; }

  friend constexpr auto operator==(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr auto operator!=(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ != rhs.value_;
  }

  friend auto CarbonHash(HashCode code) -> HashCode { return code; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x16}", value_);
  }

 private:
  uint64_t value_ = 0;
};

// Computes a hash code for the provided value, incorporating the provided seed.
//
// The seed doesn't need to be of any particular high quality, but a zero seed
// has bad effects in several places. Prefer the unseeded routine rather than
// providing a zero here.
//
// This **not** a cryptographically secure or stable hash -- it is only designed
// for use with in-memory hash table style data structures. Being fast and
// effective for that use case is the guiding principle of its design.
//
// There is no guarantee that the values produced are stable from execution to
// execution. For speed and quality reasons, the implementation does not
// introduce any variance to defend against accidental dependencies. As a
// consequence, it is strongly encouraged to use a seed that varies from
// execution to execution to avoid depending on specific values produced.
//
// The algorithm used is most heavily based on [Abseil's hashing algorithm][1],
// with some additional ideas and inspiration from the fallback hashing
// algorithm in [Rust's AHash][2]. However, there are also *significant* changes
// introduced here.
//
// [1]: https://github.com/abseil/abseil-cpp/tree/master/absl/hash/internal
// [2]: https://github.com/tkaitchuck/aHash/wiki/AHash-fallback-algorithm
//
// This hash algorithm does *not* defend against hash flooding. While it can be
// viewed as "keyed" on the seed, it is expected to be possible to craft inputs
// for some data types that cancel out the seed used and manufacture endlessly
// colliding sets of keys. In general, this function works to be *fast* for hash
// tables. If you need to defend against hash flooding, either directly use a
// data structure with strong worst-case guarantees, or a hash table which
// detects catastrophic collisions and falls back to such a data structure.
//
// This hash function is heavily optimized for *latency* over *quality*. Modern
// hash tables designs can efficiently handle reasonable collision rates,
// including using extra bits from the hash to avoid all efficiency coming from
// the same low bits. Because of this, low-latency is significantly more
// important for performance than high-quality, and this is heavily leveraged.
// The result is that the hash codes produced *do* have significant avalanche
// problems for small keys. The upside is that the latency for hashing integers,
// pointers, and small byte strings (up to 32-bytes) is exceptionally low, and
// essentially a small constant time instruction sequence. Especially for short
// strings, this function is often significantly faster even than Abseil's hash
// function or any other we are aware of. Longer byte strings are reasonably
// fast as well, competitive or better than Abseil's hash function.
//
// No exotic instruction set extensions are required, and the state used is
// small. It does rely on being able to get the low- and high-64-bit results of
// a 64-bit multiply efficiently.
//
// The function supports many typical data types such as primitives, string-ish
// views, and types composing primitives transparently like pairs, tuples, and
// array-ish views. It is also extensible to support user-defined types.
//
// To add support for your type, you need to implement a customization point --
// a free function that can be found by ADL for your type -- called `CarbonHash`
// with the following signature:
//
// ```cpp
// auto CarbonHash(Hasher hasher, const YourType& value) -> Hasher;
// ```
//
// This function needs to ensure that values that compare the same (including
// any comparisons with different types that might be used with a hash table of
// `YourType` keys) have the exact same updates to the `Hasher` object. The
// `Hasher` object should be updated with enough state so that values that
// compare unequal of your type (or other types that might be compared) have a
// high probability of different hashes. Typically this involves updating all of
// the salient state of your type into the `Hasher`. The updated hasher should
// be returned from the function.
//
// See the comments on the `Hasher` type for more details about implementing
// these customization points and how best to incorporate state into the hasher.
template <typename T>
inline auto HashValue(const T& value, uint64_t seed) -> HashCode;

// The same as the seeded version of `HashValue` but without callers needing to
// provide a seed.
//
// Generally prefer the seeded version, but this is available if there is no
// reasonable seed. In particular, this will behave better than using a seed of
// `0`.
template <typename T>
inline auto HashValue(const T& value) -> HashCode;

// Object and APIs that eventually produce a hash code.
//
// This type is primarily used by types to implement a customization point
// `CarbonHash` that will in turn be used by the `HashValue` function. See the
// `HashValue` function for details of that extension point.
//
// The API of this type can be used to incorporate data from your user-defined
// type into a hash code. The API is a low-level collection of primitives rather
// than a higher-level abstraction. This reflects a fundamental priority: deep,
// detailed, and latency-oriented optimization of any hash functions. This
// type's API also reflects the reality that high-performance hash tables rely
// on keys that are generally small and cheap to hash. The result is
// prioritizing hashing a small number of integers (1 or 2 ideally), or some
// contiguous byte buffer. We typically want hash tables whose key types work to
// fit into this model rather than advanced or more complex hashing of more
// complex input data.
//
// It also results in an unusual API design where all the functions are actually
// *static* member functions that expect the `Hasher` and its state to be moved
// into the first parameter and returned, updated, by value. This is in contrast
// to non-static member functions in C++ which would mutate the state in place
// by taking its address for the implicit `this` parameter. While that *usually*
// optimizes away, it creates a more difficult optimization environment that has
// resulted in performance issues over time and at scale.
//
// Another important note is that types only need to include the data that would
// lead to an object comparing equal or not-equal to some other object,
// including objects of other types if a type supports heterogeneous equality
// comparison. Note that only truly transparent types can be used
// heterogeneously with this hash function such as strings and string views, or
// vectors and array refs. A notable counter example are signed integers -- in
// order to make hashing of them efficient the hash will be different for
// different bit-widths, preventing heterogeneous lookup along that axis.
//
// To illustrate this -- if a type has some fixed amount of data, maybe 32
// 4-byte integers, and equality comparison only operates on the *values* of
// those integers, then the size (32 integers, or 128 bytes) doesn't need to be
// included in the hash. But if a type has a *dynamic* amount of data, and the
// sizes are compared as part of equality comparison, then that dynamic size
// should typically be included in the hash.
//
// It is essential that values that compare equal have equal hashes, and
// desirable that values that compare unequal have a high probability of
// different hashes.
class Hasher {
 public:
  Hasher(Hasher&& arg) = default;
  Hasher(const Hasher& arg) = delete;
  auto operator=(Hasher&& rhs) -> Hasher& = default;

  // Extracts the current state as a `HashCode` for use.
  explicit operator HashCode() const { return HashCode(buffer); }

  // Incorporates an object into the `hasher`s state by hashing its object
  // representation, and returns the updated `hasher`. Requires `value`'s type
  // to have a unique object representation. This is primarily useful for
  // builtin and primitive types.
  //
  // This can be directly used for simple users combining some aggregation of
  // objects. However, when possible, prefer the variadic version below for
  // aggregating several primitive types into a hash.
  template <typename T, typename = std::enable_if_t<
                            std::has_unique_object_representations_v<T>>>
  static auto Hash(Hasher hasher, const T& value) -> Hasher;

  // Incorporates a variable number of objects into the `hasher`s state in a
  // similar manner to applying the above function to each one in series. It has
  // the same requirements as the above function fer each `value`. And it
  // returns the updated `hasher`.
  //
  // There is no guaranteed correspondence between the behavior of a single call
  // with multiple parameters and multiple calls. This routine is also optimized
  // for handling relatively small numbers of objects. For hashing large
  // aggregations, consider some Merkle-tree decomposition or arranging for a
  // byte buffer that can be hashed as a single buffer. However, hashing large
  // aggregations of data in this way is rarely results in effectively
  // high-performance hash table data structures and so should generally be
  // avoided.
  template <typename... Ts,
            typename = std::enable_if_t<
                (... && std::has_unique_object_representations_v<Ts>)>>
  static auto Hash(Hasher hasher, const Ts&... value) -> Hasher;

  // Simpler and more primitive functions to incorporate state represented in
  // `uint64_t` values into the `hasher` state. The updated `hasher` is
  // returned.
  static auto HashOne(Hasher hasher, uint64_t data) -> Hasher;
  static auto HashTwo(Hasher hasher, uint64_t data0, uint64_t data1) -> Hasher;

  // A heavily optimized routine for incorporating a dynamically sized sequence
  // of bytes into `hasher`s state. The updated state is returned.
  //
  // This routine has carefully structured inline code paths for short byte
  // sequences and a reasonably high bandwidth code path for longer sequences.
  // The size of the byte sequence is always incorporated into the hasher's
  // state along with the contents.
  static auto HashSizedBytes(Hasher hasher, llvm::ArrayRef<std::byte> bytes)
      -> Hasher;

  // Read data of various sizes efficiently into one or two 64-bit values. These
  // pointers need-not be aligned, and can alias other objects. The
  // representation of the read data in the `uint64_t` returned is not stable or
  // guaranteed.
  static auto Read1(const std::byte* data) -> uint64_t;
  static auto Read2(const std::byte* data) -> uint64_t;
  static auto Read4(const std::byte* data) -> uint64_t;
  static auto Read8(const std::byte* data) -> uint64_t;
  static auto Read1To3(const std::byte* data, ssize_t size) -> uint64_t;
  static auto Read4To8(const std::byte* data, ssize_t size) -> uint64_t;
  static auto Read8To16(const std::byte* data, ssize_t size)
      -> std::pair<uint64_t, uint64_t>;

  // Reads the underlying object representation of a type into a 64-bit integer
  // efficiently. Only supports types with unique object representation and at
  // most 8-bytes large. This is typically used to read primitive types.
  template <typename T,
            typename = std::enable_if_t<
                std::has_unique_object_representations_v<T> && sizeof(T) <= 8>>
  static auto ReadSmall(const T& value) -> uint64_t;

  // The core of the hash algorithm is this mix function. The specific
  // operations are not guaranteed to be stable but are described here for
  // hashing authors to understand what to expect.
  //
  // Currently, this uses the same "mix" operation as in Abseil, AHash, and
  // several other hashing algorithms. It takes two 64-bit integers, and
  // multiplies them, capturing both the high 64-bit result and the low 64-bit
  // result, and then XOR-ing those two halves together.
  //
  // A consequence of this operation is that a zero on either side will fail to
  // incorporate any bits from the other side. Often, this is an acceptable rate
  // of collision in practice. But it is worth being aware of and working to
  // avoid common paths encountering this. For example, naively used this might
  // cause different length all-zero byte strings to hash the same, essentially
  // losing the length in the composition of the hash for a likely important
  // case of byte sequence.
  //
  // Another consequence of the particular implementation is that it is useful
  // to have a reasonable distribution of bits throughout both sides of the
  // multiplication. However, it is not *necessary* as we do capture the
  // complete 128-bit result. Where reasonable, the caller should XOR random
  // data into operands before calling `Mix` to try and increase the
  // distribution of bits feeding the multiply.
  static auto Mix(uint64_t lhs, uint64_t rhs) -> uint64_t;

  // We have a 64-byte random data pool designed to fit on a single cache line.
  // This routine allows sampling it at byte indices, which allows getting 64 -
  // 8 different random 64-bit results. The offset must be in the range [0, 56).
  static auto SampleRandomData(ssize_t offset) -> uint64_t {
    CARBON_DCHECK(offset + sizeof(uint64_t) < sizeof(StaticRandomData));
    uint64_t data;
    memcpy(&data,
           reinterpret_cast<const unsigned char*>(&StaticRandomData) + offset,
           sizeof(data));
    return data;
  }

  // A throughput-optimized routine for when the byte sequence size is
  // guaranteed to be >32.
  static auto HashSizedBytesLarge(Hasher hasher,
                                  llvm::ArrayRef<std::byte> bytes) -> Hasher;

  // Random data taken from the hexadecimal digits of Pi's fractional component,
  // written in lexical order for convenience of reading. The resulting
  // byte-stream will be different due to little-endian integers. These can be
  // used directly for convenience rather than calling `SampleRandomData`, but
  // be aware that this is the underlying pool. The goal is to reuse the same
  // single cache-line of constant data.
  //
  // The initializers here can be generated with the following shell script,
  // which will generate 8 64-bit values and one more digit. The `bc` command's
  // decimal based scaling means that without getting at least some extra hex
  // digits rendered there will be rounding that we don't want so the script
  // below goes on to produce one more hex digit ensuring the the 8 initializers
  // aren't rounded in any way. Using a higher scale won't cause the 8
  // initializers here to change further.
  //
  // ```sh
  // echo 'obase=16; scale=155; 4*a(1)' | env BC_LINE_LENGTH=500 bc -l \
  //  | cut -c 3- | tr '[:upper:]' '[:lower:]' \
  //  | sed -e "s/.\{4\}/&'/g" \
  //  | sed -e "s/\(.\{4\}'.\{4\}'.\{4\}'.\{4\}\)'/0x\1,\n/g"
  // ```
  static inline constexpr std::array<uint64_t, 8> StaticRandomData = {
      0x243f'6a88'85a3'08d3, 0x1319'8a2e'0370'7344, 0xa409'3822'299f'31d0,
      0x082e'fa98'ec4e'6c89, 0x4528'21e6'38d0'1377, 0xbe54'66cf'34e9'0c6c,
      0xc0ac'29b7'c97c'50dd, 0x3f84'd5b5'b547'0917,
  };

 private:
  template <typename T>
  friend auto HashValue(const T& value, uint64_t seed) -> HashCode;
  template <typename T>
  friend auto HashValue(const T& value) -> HashCode;

  explicit Hasher(uint64_t seed) : buffer(seed) {}

  Hasher() = default;

  // The multiplicative hash constant from Knuth, derived from 2^64 / Phi.
  static constexpr uint64_t MulConstant = 0x9e37'79b9'7f4a'7c15U;

  uint64_t buffer;
};

// A dedicated namespace for `CarbonHash` overloads that are not found by ADL
// with their associated types. For example, primitive type overloads or
// overloads for types in LLVM's libraries.
namespace HashDispatch {

inline auto CarbonHash(Hasher hasher, llvm::ArrayRef<std::byte> bytes)
    -> Hasher {
  hasher = Hasher::HashSizedBytes(std::move(hasher), bytes);
  return hasher;
}

inline auto CarbonHash(Hasher hasher, llvm::StringRef value) -> Hasher {
  return CarbonHash(
      std::move(hasher),
      llvm::ArrayRef<std::byte>(
          reinterpret_cast<const std::byte*>(value.data()), value.size()));
}

inline auto CarbonHash(Hasher hasher, std::string_view value) -> Hasher {
  return CarbonHash(
      std::move(hasher),
      llvm::ArrayRef<std::byte>(
          reinterpret_cast<const std::byte*>(value.data()), value.size()));
}

inline auto CarbonHash(Hasher hasher, const std::string& value) -> Hasher {
  return CarbonHash(
      std::move(hasher),
      llvm::ArrayRef<std::byte>(
          reinterpret_cast<const std::byte*>(value.data()), value.size()));
}

// C++ guarantees this is true for the unsigned variants, but we require it for
// signed variants and pointers.
static_assert(std::has_unique_object_representations_v<int8_t>);
static_assert(std::has_unique_object_representations_v<int16_t>);
static_assert(std::has_unique_object_representations_v<int32_t>);
static_assert(std::has_unique_object_representations_v<int64_t>);
static_assert(std::has_unique_object_representations_v<void*>);

// C++ uses `std::nullptr_t` but unfortunately doesn't make it have a unique
// object representation. To address that, we need a function that converts
// `nullptr` back into a `void*` that will have a unique object representation.
// And this needs to be done by-value as we need to build a temporary object to
// return, which requires a separate overload rather than just using a type
// function that could be used in parallel in the predicate below. Instead, we
// build the predicate independently of the mapping overload, but together they
// should produce the correct result.
template <typename T>
inline auto MapNullPtrToVoidPtr(const T& value) -> const T& {
  // This overload should never be selected for `std::nullptr_t`, so
  // static_assert to get some better compiler error messages.
  static_assert(!std::is_same_v<T, std::nullptr_t>);
  return value;
}
inline auto MapNullPtrToVoidPtr(std::nullptr_t /*value*/) -> const void* {
  return nullptr;
}

// Predicate to be used in conjunction with a `nullptr` mapping routine like the
// above.
template <typename T>
constexpr bool NullPtrOrHasUniqueObjectRepresentations =
    std::is_same_v<T, std::nullptr_t> ||
    std::has_unique_object_representations_v<T>;

template <typename T, typename = std::enable_if_t<
                          NullPtrOrHasUniqueObjectRepresentations<T>>>
inline auto CarbonHash(Hasher hasher, const T& value) -> Hasher {
  return Hasher::Hash(std::move(hasher), MapNullPtrToVoidPtr(value));
}

template <typename... Ts,
          typename = std::enable_if_t<
              (... && NullPtrOrHasUniqueObjectRepresentations<Ts>)>>
inline auto CarbonHash(Hasher hasher, const std::tuple<Ts...>& value)
    -> Hasher {
  return std::apply(
      [&](const auto&... args) {
        return Hasher::Hash(std::move(hasher), MapNullPtrToVoidPtr(args)...);
      },
      value);
}

template <typename T, typename U,
          typename = std::enable_if_t<
              NullPtrOrHasUniqueObjectRepresentations<T> &&
              NullPtrOrHasUniqueObjectRepresentations<U> &&
              sizeof(T) <= sizeof(uint64_t) && sizeof(U) <= sizeof(uint64_t)>>
inline auto CarbonHash(Hasher hasher, const std::pair<T, U>& value) -> Hasher {
  return CarbonHash(std::move(hasher), std::tuple(value.first, value.second));
}

template <typename T, typename = std::enable_if_t<
                          std::has_unique_object_representations_v<T>>>
inline auto CarbonHash(Hasher hasher, llvm::ArrayRef<T> objs) -> Hasher {
  return CarbonHash(
      std::move(hasher),
      llvm::ArrayRef(reinterpret_cast<const std::byte*>(objs.data()),
                     objs.size() * sizeof(T)));
}

template <typename T>
inline auto DispatchImpl(Hasher hasher, const T& value) -> Hasher {
  // This unqualified call will find both the overloads in this namespace and
  // ADL-found functions in an associated namespace of `T`.
  return CarbonHash(std::move(hasher), value);
}

}  // namespace HashDispatch

template <typename T>
inline auto HashValue(const T& value, uint64_t seed) -> HashCode {
  return static_cast<HashCode>(HashDispatch::DispatchImpl(Hasher(seed), value));
}

template <typename T>
inline auto HashValue(const T& value) -> HashCode {
  // When a seed isn't provided, use the last 64-bit chunk of random data. Other
  // chunks (especially the first) are more often XOR-ed with the seed and risk
  // cancelling each other out and feeding a zero to a `Mix` call in a way that
  // sharply increasing collisions.
  return HashValue(value, Hasher::StaticRandomData[7]);
}

inline auto Hasher::Read1(const std::byte* data) -> uint64_t {
  uint8_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Hasher::Read2(const std::byte* data) -> uint64_t {
  uint16_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Hasher::Read4(const std::byte* data) -> uint64_t {
  uint32_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Hasher::Read8(const std::byte* data) -> uint64_t {
  uint64_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Hasher::Read1To3(const std::byte* data, ssize_t size) -> uint64_t {
  // Use carefully crafted indexing to avoid branches on the exact size while
  // reading.
  uint64_t byte0 = static_cast<uint8_t>(data[0]);
  uint64_t byte1 = static_cast<uint8_t>(data[size - 1]);
  uint64_t byte2 = static_cast<uint8_t>(data[size / 2]);
  return byte0 | (byte1 << 16) | (byte2 << 8);
}

inline auto Hasher::Read4To8(const std::byte* data, ssize_t size) -> uint64_t {
  uint32_t low;
  std::memcpy(&low, data, sizeof(low));
  uint32_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return low | (static_cast<uint64_t>(high) << 32);
}

inline auto Hasher::Read8To16(const std::byte* data, ssize_t size)
    -> std::pair<uint64_t, uint64_t> {
  uint64_t low;
  std::memcpy(&low, data, sizeof(low));
  uint64_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return {low, high};
}

inline auto Hasher::Mix(uint64_t lhs, uint64_t rhs) -> uint64_t {
  // Use the C23 extended integer support that Clang provides as a general
  // language extension.
  using U128 = unsigned _BitInt(128);
  U128 result = static_cast<U128>(lhs) * static_cast<U128>(rhs);
  return static_cast<uint64_t>(result) ^ static_cast<uint64_t>(result >> 64);
}

inline auto Hasher::HashOne(Hasher hasher, uint64_t data) -> Hasher {
  // When hashing exactly one 64-bit entity use the Phi-derived constant as this
  // is just multiplicative hashing. The initial buffer is mixed on input to
  // pipeline with materializing the constant.
  hasher.buffer = Mix(data ^ hasher.buffer, MulConstant);
  return hasher;
}

inline auto Hasher::HashTwo(Hasher hasher, uint64_t data0, uint64_t data1)
    -> Hasher {
  // When hashing two chunks of data at the same time, we XOR it with random
  // data to avoid common inputs from having especially bad multiplicative
  // effects. We also XOR in the starting buffer as seed or to chain. Note that
  // we don't use *consecutive* random data 64-bit values to avoid a common
  // compiler "optimization" of loading both 64-bit chunks into a 128-bit vector
  // and doing the XOR in the vector unit. The latency of extracting the data
  // afterward eclipses any benefit. Callers will routinely have two consecutive
  // data values here, but using non-consecutive keys avoids any vectorization
  // being tempting.
  //
  // XOR-ing both the incoming state and a random word over the second data is
  // done to pipeline with materializing the constants and is observed to have
  // better performance than XOR-ing after the mix.
  //
  // This roughly matches the mix pattern used in the larger mixing routines
  // from Abseil, which is a more minimal form than used in other algorithms
  // such as AHash and seems adequate for latency-optimized use cases.
  hasher.buffer = Mix(data0 ^ StaticRandomData[1],
                      data1 ^ StaticRandomData[3] ^ hasher.buffer);
  return hasher;
}

inline auto Hasher::HashSizedBytes(Hasher hasher,
                                   llvm::ArrayRef<std::byte> bytes) -> Hasher {
  const std::byte* data_ptr = bytes.data();
  const ssize_t size = bytes.size();

  // First handle short sequences under 8 bytes.
  if (LLVM_UNLIKELY(size == 0)) {
    hasher = HashOne(std::move(hasher), 0);
    return hasher;
  }
  if (size <= 8) {
    uint64_t data;
    if (size >= 4) {
      data = Read4To8(data_ptr, size);
    } else {
      data = Read1To3(data_ptr, size);
    }
    // We optimize for latency on short strings by hashing both the data and
    // size in a single multiply here. This results in a *statistically* weak
    // hash function. It would be improved by doing two rounds of multiplicative
    // hashing which is what many other modern multiplicative hashes do,
    // including Abseil and others:
    //
    // ```cpp
    // hash = HashOne(std::move(hash), data);
    // hash = HashOne(std::move(hash), size);
    // ```
    //
    // We opt to make the same tradeoff here for small sized strings that both
    // this library and Abseil make for *fixed* size integers by using a weaker
    // single round of multiplicative hashing and a size-dependent constant
    // loaded from memory.
    hasher.buffer = Mix(data ^ hasher.buffer, SampleRandomData(size));
    return hasher;
  }

  if (size <= 16) {
    // Similar to the above, we optimize primarily for latency here and spread
    // the incoming data across both ends of the multiply. Note that this does
    // have a drawback -- any time one half of the mix function becomes zero it
    // will fail to incorporate any bits from the other half. However, there is
    // exactly 1 in 2^64 values for each side that achieve this, and only when
    // the size is exactly 16 -- for smaller sizes there is an overlapping byte
    // that makes this impossible unless the seed is *also* incredibly unlucky.
    //
    // Because this hash function makes no attempt to defend against hash
    // flooding, we accept this risk in order to keep the latency low. If this
    // becomes a non-flooding problem, we can restrict the size to <16 and send
    // the 16-byte case down the next tier of cost.
    auto data = Read8To16(data_ptr, size);
    hasher.buffer =
        Mix(data.first ^ SampleRandomData(size), data.second ^ hasher.buffer);
    return hasher;
  }

  if (size <= 32) {
    // Do two mixes of overlapping 16-byte ranges in parallel to minimize
    // latency. We also incorporate the size by sampling random data into the
    // seed before both.
    hasher.buffer ^= SampleRandomData(size);
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ hasher.buffer);

    const std::byte* tail_16b_ptr = data_ptr + (size - 16);
    uint64_t m1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[3],
                      Read8(tail_16b_ptr + 8) ^ hasher.buffer);
    // Just an XOR mix at the end is quite weak here, but we prefer that for
    // latency over a more robust approach. Doing another mix with the size (the
    // way longer string hashing does) increases the latency on x86-64
    // significantly (approx. 20%).
    hasher.buffer = m0 ^ m1;
    return hasher;
  }

  return HashSizedBytesLarge(std::move(hasher), bytes);
}

template <typename T, typename /*enable_if*/>
inline auto Hasher::ReadSmall(const T& value) -> uint64_t {
  const auto* storage = reinterpret_cast<const std::byte*>(&value);
  if constexpr (sizeof(T) == 1) {
    return Read1(storage);
  } else if constexpr (sizeof(T) == 2) {
    return Read2(storage);
  } else if constexpr (sizeof(T) == 3) {
    return Read2(storage) | (Read1(&storage[2]) << 16);
  } else if constexpr (sizeof(T) == 4) {
    return Read4(storage);
  } else if constexpr (sizeof(T) == 5) {
    return Read4(storage) | (Read1(&storage[4]) << 32);
  } else if constexpr (sizeof(T) == 6 || sizeof(T) == 7) {
    // Use overlapping 4-byte reads for 6 and 7 bytes.
    return Read4(storage) | (Read4(&storage[sizeof(T) - 4]) << 32);
  } else if constexpr (sizeof(T) == 8) {
    return Read8(storage);
  } else {
    static_assert(sizeof(T) <= 8);
  }
}

template <typename T, typename /*enable_if*/>
inline auto Hasher::Hash(Hasher hasher, const T& value) -> Hasher {
#if 0
  // For integer types up to 64-bit widths, we hash the 2's compliment value by
  // casting to unsigned and hashing as a `uint64_t`. This has the downside of
  // making negative integers hash differently based on their width, but
  // anything else would require potentially expensive sign extension. For
  // positive integers though, there is no problem with using differently sized
  // integers (for example literals) than the stored keys.
  if constexpr (sizeof(T) <= 8 &&
                (std::is_enum_v<T> || std::is_integral_v<T>)) {
    uint64_t ext_value = static_cast<std::make_unsigned_t<T>>(value);
    return HashOne(std::move(hasher), ext_value);
  }
#else
  if constexpr (sizeof(T) <= 4) {
    // Get any entropy bits from a pointer seed into the low bits. Harmless if
    // the seed isn't a pointer, and should fit into the XOR instruction on Arm
    // and pipeline with the multiply even on x86.
    hasher.buffer = llvm::rotr(hasher.buffer, 12);
    hasher.buffer ^= static_cast<uint32_t>(ReadSmall(value)) * MulConstant;
    return hasher;
  }
#endif

  // We don't need the size to be part of the hash, as the size here is just a
  // function of the type and we're hashing to distinguish different values of
  // the same type. So we just dispatch to the fastest path for the specific
  // size in question.
  if constexpr (sizeof(T) <= 8) {
    hasher = HashOne(std::move(hasher), ReadSmall(value));
    return hasher;
  }

  const auto* data_ptr = reinterpret_cast<const std::byte*>(&value);
  if constexpr (8 < sizeof(T) && sizeof(T) <= 16) {
    auto values = Read8To16(data_ptr, sizeof(T));
    hasher = HashTwo(std::move(hasher), values.first, values.second);
    return hasher;
  }

  if constexpr (16 < sizeof(T) && sizeof(T) <= 32) {
    // Essentially the same technique used for dynamically sized byte sequences
    // of this size, but we start with a fixed XOR of random data.
    hasher.buffer ^= StaticRandomData[0];
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ hasher.buffer);
    const std::byte* tail_16b_ptr = data_ptr + (sizeof(T) - 16);
    uint64_t m1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[3],
                      Read8(tail_16b_ptr + 8) ^ hasher.buffer);
    hasher.buffer = m0 ^ m1;
    return hasher;
  }

  // Hashing the size isn't relevant here, but is harmless, so fall back to a
  // common code path.
  return HashSizedBytesLarge(std::move(hasher),
                             llvm::ArrayRef<std::byte>(data_ptr, sizeof(T)));
}

template <typename... Ts, typename /*enable_if*/>
inline auto Hasher::Hash(Hasher hasher, const Ts&... value) -> Hasher {
  if constexpr (sizeof...(Ts) == 0) {
    return HashOne(std::move(hasher), 0);
  }
  if constexpr (sizeof...(Ts) == 1) {
    return Hash(std::move(hasher), value...);
  }
  if constexpr ((... && (sizeof(Ts) <= 8))) {
    if constexpr (sizeof...(Ts) == 2) {
      return HashTwo(std::move(hasher), ReadSmall(value)...);
    }

    // More than two, but all small -- read each one into a contiguous buffer of
    // data. This may be a bit memory wasteful by padding everything out to
    // 8-byte chunks, but for that regularity the hashing is likely faster.
    const uint64_t data[] = {ReadSmall(value)...};
    return Hash(std::move(hasher), data);
  }

  // For larger objects, hash each one down to a hash code and then hash those
  // as a buffer.
  const uint64_t data[] = {static_cast<uint64_t>(
      static_cast<HashCode>(Hash(Hasher(hasher.buffer), value)))...};
  return Hash(std::move(hasher), data);
}

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHING_H_
