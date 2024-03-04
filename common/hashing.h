// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_HASHING_H_
#define CARBON_COMMON_HASHING_H_

#include <concepts>
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

#ifdef __ARM_ACLE
#include <arm_acle.h>
#endif

namespace Carbon {

// A 64-bit hash code produced by `Carbon::HashValue`.
//
// This provides methods for extracting high-quality bits from the hash code
// quickly.
//
// This class can also be a hashing input when recursively hashing more complex
// data structures.
class HashCode : public Printable<HashCode> {
 public:
  HashCode() = default;

  constexpr explicit HashCode(uint64_t value) : value_(value) {}

  friend constexpr auto operator==(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr auto operator!=(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ != rhs.value_;
  }

  // Extracts an index from the hash code as a `ssize_t`. This index covers the
  // full range of that type, and may even be negative. Typical usage will
  // involve masking this down to some positive range using a bitand with a mask
  // computed from a power-of-two size. This routine doesn't do any masking to
  // ensure a positive index to avoid redundant computations with the typical
  // user of the index.
  constexpr auto ExtractIndex() -> ssize_t;

  // Extracts an index and a fixed `N`-bit tag from the hash code.
  //
  // This extracts these values from the position of the hash code which
  // maximizes the entropy in the tag and the low bits of the index, as typical
  // indices will be further masked down to fall in a smaller range.
  //
  // `N` must be in the range [1, 32]. The returned index will be in the range
  // [0, 2**(64-N)).
  template <int N>
  constexpr auto ExtractIndexAndTag() -> std::pair<ssize_t, uint32_t>;

  // Extract the full 64-bit hash code as an integer.
  //
  // The methods above should be preferred rather than directly manipulating
  // this integer. This is provided primarily to enable Merkle-tree hashing or
  // other recursive hashing where that is needed or more efficient.
  explicit operator uint64_t() const { return value_; }

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
// algorithm in [Rust's AHash][2] and the [FxHash][3] function. However, there
// are also *significant* changes introduced here.
//
// [1]: https://github.com/abseil/abseil-cpp/tree/master/absl/hash/internal
// [2]: https://github.com/tkaitchuck/aHash/wiki/AHash-fallback-algorithm
// [3]: https://docs.rs/fxhash/latest/fxhash/
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
// essentially a small constant time instruction sequence.
//
// No exotic instruction set extensions are required, and the state used is
// small. It does rely on being able to get the low- and high-64-bit results of
// a 64-bit multiply efficiently.
//
// The function supports many typical data types such as primitives, string-ish
// views, and types composing primitives transparently like pairs, tuples, and
// array-ish views. It is also extensible to support user-defined types.
//
// The builtin support for string-like types include:
// - `std::string_view`
// - `std::string`
// - `llvm::StringRef`
// - `llvm::SmallString`
//
// This function supports heterogeneous lookup between all of the string-like
// types. It also supports heterogeneous lookup between pointer types regardless
// of pointee type and `nullptr`.
//
// However, these are the only heterogeneous lookup support including for the
// builtin in, standard, and LLVM types. Notably, each different size and
// signedness integer type may hash differently for efficiency reasons. Hash
// tables should pick a single integer type in which to manage keys and do
// lookups.
//
// To add support for your type, you need to implement a customization point --
// a free function that can be found by ADL for your type -- called
// `CarbonHashValue` with the following signature:
//
// ```cpp
// auto CarbonHashValue(const YourType& value, uint64_t seed) -> HashCode;
// ```
//
// The extension point needs to ensure that values that compare equal (including
// any comparisons with different types that might be used with a hash table of
// `YourType` keys) produce the same `HashCode` values.
//
// `HashCode` values should typically be produced using the `Hasher` helper type
// below. See its documentation for more details about implementing these
// customization points and how best to incorporate the value's state into a
// `HashCode`.
//
// For two input values that are almost but not quite equal, the extension
// point should maximize the probability of each bit of their resulting
// `HashCode`s differing. More formally, `HashCode`s should exhibit an
// [avalanche effect][4]. However, while this is desirable, it should be
// **secondary** to low latency. The intended use case of these functions is not
// cryptography but in-memory hashtables where the latency and overhead of
// computing the `HashCode` is *significantly* more important than achieving a
// particularly high quality. The goal is to have "just enough" avalanche
// effect, but there is not a fixed criteria for how much is enough. That should
// be determined through practical experimentation with a hashtable and
// distribution of keys.
//
// [4]: https://en.wikipedia.org/wiki/Avalanche_effect
template <typename T>
inline auto HashValue(const T& value, uint64_t seed) -> HashCode;

// The same as the seeded version of `HashValue` but without callers needing to
// provide a seed.
//
// Generally prefer the seeded version, but this is available if there is no
// reasonable seed. In particular, this will behave better than using a seed of
// `0`. One important use case is for recursive hashing of sub-objects where
// appropriate or needed.
template <typename T>
inline auto HashValue(const T& value) -> HashCode;

// Object and APIs that eventually produce a hash code.
//
// This type is primarily used by types to implement a customization point
// `CarbonHashValue` that will in turn be used by the `HashValue` function. See
// the `HashValue` function for details of that extension point.
//
// The methods on this type can be used to incorporate data from your
// user-defined type into its internal state which can be converted to a
// `HashCode` at any time. These methods will only produce the same `HashCode`
// if they are called in the exact same order with the same arguments -- there
// are no guaranteed equivalences between calling different methods.
//
// Example usage:
// ```cpp
// auto CarbonHashValue(const MyType& value, uint64_t seed) -> HashCode {
//   Hasher hasher(seed);
//   hasher.HashTwo(value.x, value.y);
//   return static_cast<HashCode>(hasher);
// }
// ```
//
// This type's API also reflects the reality that high-performance hash tables
// are used with keys that are generally small and cheap to hash.
//
// To ensure this type's code is optimized effectively, it should typically be
// used as a local variable and not passed across function boundaries
// unnecessarily.
//
// The type also provides a number of static helper functions and static data
// members that may be used by authors of `CarbonHashValue` implementations to
// efficiently compute the inputs to the core `Hasher` methods, or even to
// manually do some amounts of hashing in performance-tuned ways outside of the
// methods provided.
class Hasher {
 public:
  Hasher() = default;
  explicit Hasher(uint64_t seed) : buffer(seed) {}

  Hasher(Hasher&& arg) = default;
  Hasher(const Hasher& arg) = delete;
  auto operator=(Hasher&& rhs) -> Hasher& = default;

  // Extracts the current state as a `HashCode` for use.
  explicit operator HashCode() const { return HashCode(buffer); }

  // Incorporates an object into the hasher's state by hashing its object
  // representation. Requires `value`'s type to have a unique object
  // representation. This is primarily useful for builtin and primitive types.
  //
  // This can be directly used for simple users combining some aggregation of
  // objects. However, when possible, prefer the variadic version below for
  // aggregating several primitive types into a hash.
  template <typename T>
    requires std::has_unique_object_representations_v<T>
  auto Hash(const T& value) -> void;

  // Incorporates a variable number of objects into the `hasher`s state in a
  // similar manner to applying the above function to each one in series. It has
  // the same requirements as the above function for each `value`. And it
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
  template <typename... Ts>
    requires(... && std::has_unique_object_representations_v<Ts>)
  auto Hash(const Ts&... value) -> void;

  // Simpler and more primitive functions to incorporate state represented in
  // `uint64_t` values into the hasher's state.
  //
  // These may be slightly less efficient than the `Hash` method above for a
  // typical application code `uint64_t`, but are designed to work well even
  // when relevant data has been packed into the `uint64_t` parameters densely.
  auto HashDense(uint64_t data) -> void;
  auto HashDense(uint64_t data0, uint64_t data1) -> void;

  // A heavily optimized routine for incorporating a dynamically sized sequence
  // of bytes into the hasher's state.
  //
  // This routine has carefully structured inline code paths for short byte
  // sequences and a reasonably high bandwidth code path for longer sequences.
  // The size of the byte sequence is always incorporated into the hasher's
  // state along with the contents.
  auto HashSizedBytes(llvm::ArrayRef<std::byte> bytes) -> void;

  // An out-of-line, throughput-optimized routine for incorporating a
  // dynamically sized sequence when the sequence size is guaranteed to be >32.
  // The size is always incorporated into the state.
  auto HashSizedBytesLarge(llvm::ArrayRef<std::byte> bytes) -> void;

  // Utility functions to read data of various sizes efficiently into a
  // 64-bit value. These pointers need-not be aligned, and can alias other
  // objects. The representation of the read data in the `uint64_t` returned is
  // not stable or guaranteed.
  static auto Read1(const std::byte* data) -> uint64_t;
  static auto Read2(const std::byte* data) -> uint64_t;
  static auto Read4(const std::byte* data) -> uint64_t;
  static auto Read8(const std::byte* data) -> uint64_t;

  // Similar to the `ReadN` functions, but supports reading a range of different
  // bytes provided by the size *without branching on the size*. The lack of
  // branches is often key, and the code in these routines works to be efficient
  // in extracting a *dynamic* size of bytes into the returned `uint64_t`. There
  // may be overlap between different routines, because these routines are based
  // on different implementation techniques that do have some overlap in the
  // range of sizes they can support. Which routine is the most efficient for a
  // size in the overlap isn't trivial, and so these primitives are provided
  // as-is and should be selected based on the localized generated code and
  // benchmarked performance.
  static auto Read1To3(const std::byte* data, ssize_t size) -> uint64_t;
  static auto Read4To8(const std::byte* data, ssize_t size) -> uint64_t;
  static auto Read8To16(const std::byte* data, ssize_t size)
      -> std::pair<uint64_t, uint64_t>;

  // Reads the underlying object representation of a type into a 64-bit integer
  // efficiently. Only supports types with unique object representation and at
  // most 8-bytes large. This is typically used to read primitive types.
  template <typename T>
    requires std::has_unique_object_representations_v<T> && (sizeof(T) <= 8)
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

  // An alternative to `Mix` that is significantly weaker but also lower
  // latency. It should not be used when the input `uint64_t` is densely packed
  // with data, but is a good option for hashing a single integer or pointer
  // where the full 64-bits are sparsely populated and especially the high bits
  // are often invariant between interestingly different values.
  //
  // This uses just the low 64-bit result of a multiply. It ensures the operand
  // is good at diffusing bits, but inherently the high bits of the input will
  // be (significantly) less often represented in the output. It also does some
  // reversal to ensure the *low* bits of the result are the most useful ones.
  static auto WeakMix(uint64_t value) -> uint64_t;

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
  // below goes on to produce one more hex digit ensuring the 8 initializers
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

  // We need a multiplicative hashing constant for both 64-bit multiplicative
  // hashing fast paths and some other 128-bit folded multiplies. We use an
  // empirically better constant compared to Knuth's, Rust's FxHash, and others
  // we've tried. It was found by a search of uniformly distributed odd numbers
  // and examining them for desirable properties when used as a multiplicative
  // hash, however our search seems largely to have been lucky rather than
  // having a highly effective set of criteria. We evaluated this constant by
  // integrating this hash function with a hashtable and looking at the
  // collision rates of several different but very fundamental patterns of keys:
  // integers counting from 0, pointers allocated on the heap, and strings with
  // character and size distributions matching C-style ASCII identifiers.
  // Different constants found with this search worked better or less well, but
  // fairly consistently across the different types of keys. At the end, far and
  // away the best behaved constant we found was one of the first ones in the
  // search and is what we use here.
  //
  // For reference, some other constants include one derived by diving 2^64 by
  // Phi: 0x9e37'79b9'7f4a'7c15U -- see these sites for details:
  // https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
  // https://book.huihoo.com/data-structures-and-algorithms-with-object-oriented-design-patterns-in-c++/html/page214.html
  //
  // Another very good constant derived by minimizing repeating bit patterns is
  // 0xdcb2'2ca6'8cb1'34edU and its bit-reversed form. However, this constant
  // has observed frequent issues at roughly 4k pointer keys, connected to a
  // common hashtable seed also being a pointer. These issues appear to occur
  // both more often and have a larger impact relative to the number of keys
  // than the rare cases where some combinations of pointer seeds and pointer
  // keys create minor quality issues with the constant we use.
  static constexpr uint64_t MulConstant = 0x7924'f9e0'de1e'8cf5U;

 private:
  uint64_t buffer;
};

// A dedicated namespace for `CarbonHashValue` overloads that are not found by
// ADL with their associated types. For example, primitive type overloads or
// overloads for types in LLVM's libraries.
//
// Note that these are internal implementation details and **not** part of the
// public API. They should not be used directly by client code.
namespace InternalHashDispatch {

inline auto CarbonHashValue(llvm::ArrayRef<std::byte> bytes, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashSizedBytes(bytes);
  return static_cast<HashCode>(hasher);
}

// Hashing implementation for `llvm::StringRef`. We forward all the other
// string-like types that support heterogeneous lookup to this one.
inline auto CarbonHashValue(llvm::StringRef value, uint64_t seed) -> HashCode {
  return CarbonHashValue(
      llvm::ArrayRef(reinterpret_cast<const std::byte*>(value.data()),
                     value.size()),
      seed);
}

inline auto CarbonHashValue(std::string_view value, uint64_t seed) -> HashCode {
  return CarbonHashValue(llvm::StringRef(value.data(), value.size()), seed);
}

inline auto CarbonHashValue(const std::string& value, uint64_t seed)
    -> HashCode {
  return CarbonHashValue(llvm::StringRef(value.data(), value.size()), seed);
}

template <unsigned Length>
inline auto CarbonHashValue(const llvm::SmallString<Length>& value,
                            uint64_t seed) -> HashCode {
  return CarbonHashValue(llvm::StringRef(value.data(), value.size()), seed);
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
  static_assert(!std::same_as<T, std::nullptr_t>);
  return value;
}
inline auto MapNullPtrToVoidPtr(std::nullptr_t /*value*/) -> const void* {
  return nullptr;
}

// Implementation detail predicate to be used in conjunction with a `nullptr`
// mapping routine like the above.
template <typename T>
concept NullPtrOrHasUniqueObjectRepresentations =
    std::same_as<T, std::nullptr_t> ||
    std::has_unique_object_representations_v<T>;

template <typename T>
  requires NullPtrOrHasUniqueObjectRepresentations<T>
inline auto CarbonHashValue(const T& value, uint64_t seed) -> HashCode {
  Hasher hasher(seed);
  hasher.Hash(MapNullPtrToVoidPtr(value));
  return static_cast<HashCode>(hasher);
}

template <typename... Ts>
  requires(... && NullPtrOrHasUniqueObjectRepresentations<Ts>)
inline auto CarbonHashValue(const std::tuple<Ts...>& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  std::apply(
      [&](const auto&... args) { hasher.Hash(MapNullPtrToVoidPtr(args)...); },
      value);
  return static_cast<HashCode>(hasher);
}

template <typename T, typename U>
  requires NullPtrOrHasUniqueObjectRepresentations<T> &&
           NullPtrOrHasUniqueObjectRepresentations<U> &&
           (sizeof(T) <= sizeof(uint64_t) && sizeof(U) <= sizeof(uint64_t))
inline auto CarbonHashValue(const std::pair<T, U>& value, uint64_t seed)
    -> HashCode {
  return CarbonHashValue(std::tuple(value.first, value.second), seed);
}

template <typename T>
  requires std::has_unique_object_representations_v<T>
inline auto CarbonHashValue(llvm::ArrayRef<T> objs, uint64_t seed) -> HashCode {
  return CarbonHashValue(
      llvm::ArrayRef(reinterpret_cast<const std::byte*>(objs.data()),
                     objs.size() * sizeof(T)),
      seed);
}

template <typename T>
inline auto DispatchImpl(const T& value, uint64_t seed) -> HashCode {
  // This unqualified call will find both the overloads in this namespace and
  // ADL-found functions in an associated namespace of `T`.
  return CarbonHashValue(value, seed);
}

}  // namespace InternalHashDispatch

template <typename T>
inline auto HashValue(const T& value, uint64_t seed) -> HashCode {
  return InternalHashDispatch::DispatchImpl(value, seed);
}

template <typename T>
inline auto HashValue(const T& value) -> HashCode {
  // When a seed isn't provided, use the last 64-bit chunk of random data. Other
  // chunks (especially the first) are more often XOR-ed with the seed and risk
  // cancelling each other out and feeding a zero to a `Mix` call in a way that
  // sharply increasing collisions.
  return HashValue(value, Hasher::StaticRandomData[7]);
}

inline constexpr auto HashCode::ExtractIndex() -> ssize_t { return value_; }

template <int N>
inline constexpr auto HashCode::ExtractIndexAndTag()
    -> std::pair<ssize_t, uint32_t> {
  static_assert(N >= 1);
  static_assert(N <= 32);
  return {static_cast<ssize_t>(value_ >> N),
          static_cast<uint32_t>(value_ & ((1U << (N + 1)) - 1))};
}

// Building with `-DCARBON_MCA_MARKERS` will enable `llvm-mca` annotations in
// the source code. These can interfere with optimization, but allows analyzing
// the generated `.s` file with the `llvm-mca` tool. Documentation for these
// markers is here:
// https://llvm.org/docs/CommandGuide/llvm-mca.html#using-markers-to-analyze-specific-code-blocks
#if CARBON_MCA_MARKERS
#define CARBON_MCA_BEGIN(NAME) \
  __asm volatile("# LLVM-MCA-BEGIN " NAME "" ::: "memory");
#define CARBON_MCA_END(NAME) \
  __asm volatile("# LLVM-MCA-END " NAME "" ::: "memory");
#else
#define CARBON_MCA_BEGIN(NAME)
#define CARBON_MCA_END(NAME)
#endif

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
  uint64_t byte2 = static_cast<uint8_t>(data[size >> 1]);
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

inline auto Hasher::WeakMix(uint64_t value) -> uint64_t {
  value *= MulConstant;
#ifdef __ARM_ACLE
  // Arm has a fast bit-reversal that gives us the optimal distribution.
  value = __rbitll(value);
#else
  // Otherwise, assume an optimized BSWAP such as x86's. That's close enough.
  value = __builtin_bswap64(value);
#endif
  return value;
}

inline auto Hasher::HashDense(uint64_t data) -> void {
  // When hashing exactly one 64-bit entity use the Phi-derived constant as this
  // is just multiplicative hashing. The initial buffer is mixed on input to
  // pipeline with materializing the constant.
  buffer = Mix(data ^ buffer, MulConstant);
}

inline auto Hasher::HashDense(uint64_t data0, uint64_t data1) -> void {
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
  buffer =
      Mix(data0 ^ StaticRandomData[1], data1 ^ StaticRandomData[3] ^ buffer);
}

template <typename T>
  requires std::has_unique_object_representations_v<T> && (sizeof(T) <= 8)
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

template <typename T>
  requires std::has_unique_object_representations_v<T>
inline auto Hasher::Hash(const T& value) -> void {
  if constexpr (sizeof(T) <= 8) {
    // For types size 8-bytes and smaller directly being hashed (as opposed to
    // 8-bytes potentially bit-packed with data), we rarely expect the incoming
    // data to fully and densely populate all 8 bytes. For these cases we have a
    // `WeakMix` routine that is lower latency but lower quality.
    CARBON_MCA_BEGIN("fixed-8b");
    buffer = WeakMix(buffer ^ ReadSmall(value));
    CARBON_MCA_END("fixed-8b");
    return;
  }

  const auto* data_ptr = reinterpret_cast<const std::byte*>(&value);
  if constexpr (8 < sizeof(T) && sizeof(T) <= 16) {
    CARBON_MCA_BEGIN("fixed-16b");
    auto values = Read8To16(data_ptr, sizeof(T));
    HashDense(values.first, values.second);
    CARBON_MCA_END("fixed-16b");
    return;
  }

  if constexpr (16 < sizeof(T) && sizeof(T) <= 32) {
    CARBON_MCA_BEGIN("fixed-32b");
    // Essentially the same technique used for dynamically sized byte sequences
    // of this size, but we start with a fixed XOR of random data.
    buffer ^= StaticRandomData[0];
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ buffer);
    const std::byte* tail_16b_ptr = data_ptr + (sizeof(T) - 16);
    uint64_t m1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[3],
                      Read8(tail_16b_ptr + 8) ^ buffer);
    buffer = m0 ^ m1;
    CARBON_MCA_END("fixed-32b");
    return;
  }

  // Hashing the size isn't relevant here, but is harmless, so fall back to a
  // common code path.
  HashSizedBytesLarge(llvm::ArrayRef<std::byte>(data_ptr, sizeof(T)));
}

template <typename... Ts>
  requires(... && std::has_unique_object_representations_v<Ts>)
inline auto Hasher::Hash(const Ts&... value) -> void {
  if constexpr (sizeof...(Ts) == 0) {
    buffer ^= StaticRandomData[0];
    return;
  }
  if constexpr (sizeof...(Ts) == 1) {
    Hash(value...);
    return;
  }
  if constexpr ((... && (sizeof(Ts) <= 8))) {
    if constexpr (sizeof...(Ts) == 2) {
      HashDense(ReadSmall(value)...);
      return;
    }

    // More than two, but all small -- read each one into a contiguous buffer of
    // data. This may be a bit memory wasteful by padding everything out to
    // 8-byte chunks, but for that regularity the hashing is likely faster.
    const uint64_t data[] = {ReadSmall(value)...};
    Hash(data);
    return;
  }

  // For larger objects, hash each one down to a hash code and then hash those
  // as a buffer.
  const uint64_t data[] = {static_cast<uint64_t>(HashValue(value))...};
  Hash(data);
}

inline auto Hasher::HashSizedBytes(llvm::ArrayRef<std::byte> bytes) -> void {
  const std::byte* data_ptr = bytes.data();
  const ssize_t size = bytes.size();

  // First handle short sequences under 8 bytes. We distribute the branches a
  // bit for short strings.
  if (size <= 8) {
    if (size >= 4) {
      CARBON_MCA_BEGIN("dynamic-8b");
      uint64_t data = Read4To8(data_ptr, size);
      // We optimize for latency on short strings by hashing both the data and
      // size in a single multiply here, using the small nature of size to
      // sample a specific sequence of bytes with well distributed bits into one
      // side of the multiply. This results in a *statistically* weak hash
      // function, but one with very low latency.
      //
      // Note that we don't drop to the `WeakMix` routine here because we want
      // to use sampled random data to encode the size, which may not be as
      // effective without the full 128-bit folded result.
      buffer = Mix(data ^ buffer, SampleRandomData(size));
      CARBON_MCA_END("dynamic-8b");
      return;
    }

    // When we only have 0-3 bytes of string, we can avoid the cost of `Mix`.
    // Instead, for empty strings we can just XOR some of our data against the
    // existing buffer. For 1-3 byte lengths we do 3 one-byte reads adjusted to
    // always read in-bounds without branching. Then we OR the size into the 4th
    // byte and use `WeakMix`.
    CARBON_MCA_BEGIN("dynamic-4b");
    if (size == 0) {
      buffer ^= StaticRandomData[0];
    } else {
      uint64_t data = Read1To3(data_ptr, size) | size << 24;
      buffer = WeakMix(data);
    }
    CARBON_MCA_END("dynamic-4b");
    return;
  }

  if (size <= 16) {
    CARBON_MCA_BEGIN("dynamic-16b");
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
    uint64_t size_hash = SampleRandomData(size);
    auto data = Read8To16(data_ptr, size);
    buffer = Mix(data.first ^ size_hash, data.second ^ buffer);
    CARBON_MCA_END("dynamic-16b");
    return;
  }

  if (size <= 32) {
    CARBON_MCA_BEGIN("dynamic-32b");
    // Do two mixes of overlapping 16-byte ranges in parallel to minimize
    // latency. We also incorporate the size by sampling random data into the
    // seed before both.
    buffer ^= SampleRandomData(size);
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ buffer);

    const std::byte* tail_16b_ptr = data_ptr + (size - 16);
    uint64_t m1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[3],
                      Read8(tail_16b_ptr + 8) ^ buffer);
    // Just an XOR mix at the end is quite weak here, but we prefer that for
    // latency over a more robust approach. Doing another mix with the size (the
    // way longer string hashing does) increases the latency on x86-64
    // significantly (approx. 20%).
    buffer = m0 ^ m1;
    CARBON_MCA_END("dynamic-32b");
    return;
  }

  HashSizedBytesLarge(bytes);
}

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHING_H_
