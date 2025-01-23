// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_INT_H_
#define CARBON_TOOLCHAIN_BASE_INT_H_

#include "common/check.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// Forward declare a testing peer so we can friend it.
namespace Testing {
struct IntStoreTestPeer;
}  // namespace Testing

// Corresponds to a canonicalized integer value. This is used both for integer
// literal tokens, and integer values in SemIR. These always represent the
// abstract mathematical value -- signed and regardless of the needed precision.
//
// Small values are internalized into the ID itself. Large values are
// represented as an index into an array of `APInt`s with a canonicalized bit
// width. The ID itself can be queried for whether it is a value-embedded-ID or
// an index ID. The ID also provides APIs for extracting either the value or an
// index.
//
// ## Details of the encoding scheme ##
//
// We need all the values from a maximum to minimum, as well as a healthy range
// of indices, to fit within the token ID bits.
//
// We represent this as a signed `TokenIdBits`-bit 2s compliment integer. The
// sign extension from TokenIdBits to a register size can be folded into the
// shift used to extract those bits from compressed bitfield storage.
//
// We then divide the smallest 1/4th of that bit width's space to represent
// indices, and the larger 3/4ths to embedded values. For 23-bits total this
// still gives us 2 million unique integers larger than the embedded ones, which
// would be difficult to fill without exceeding the number of tokens we can lex
// (8 million). For non-token based integers, the indices can continue downward
// to the 32-bit signed integer minimum, supporting approximately 1.998 billion
// unique larger integers.
//
// Note the `None` ID can't be used with a token. This is OK as we expect
// invalid tokens to be *error* tokens and not need to represent an invalid
// integer.
class IntId : public Printable<IntId> {
 public:
  static constexpr llvm::StringLiteral Label = "int";
  using ValueType = llvm::APInt;

  // The encoding of integer IDs ensures that IDs associated with tokens during
  // lexing can fit into a compressed storage space. We arrange for
  // `TokenIdBits` to be the minimum number of bits of storage for token
  // associated IDs. The constant is public so the lexer can ensure it reserves
  // adequate space.
  //
  // Note that there may still be IDs either not associated with
  // tokens or computed after lexing outside of this range.
  static constexpr int TokenIdBits = 23;

  static const IntId None;

  static auto MakeFromTokenPayload(uint32_t payload) -> IntId {
    // Token-associated IDs are signed `TokenIdBits` integers, so force sign
    // extension from that bit.
    return IntId(static_cast<int32_t>(payload << TokenIdBitsShift) >>
                 TokenIdBitsShift);
  }

  // Construct an ID from a raw 32-bit ID value.
  static constexpr auto MakeRaw(int32_t raw_id) -> IntId {
    return IntId(raw_id);
  }

  // Tests whether the ID is an embedded value ID.
  //
  // Note `None` is not an embedded value, so this implies `has_value()` is
  // true.
  constexpr auto is_embedded_value() const -> bool { return id_ > ZeroIndexId; }

  // Tests whether the ID is an index ID.
  //
  // Note `None` is represented as an index ID, so this is *not* sufficient to
  // test `has_value()`.
  constexpr auto is_index() const -> bool { return id_ <= ZeroIndexId; }

  // Test whether a value is present.
  //
  // This does not distinguish between embedded values and index IDs, only
  // whether some value is present.
  constexpr auto has_value() const -> bool { return id_ != NoneId; }

  // Converts an ID to the embedded value. Requires that `is_embedded_value()`
  // is true.
  constexpr auto AsValue() const -> int {
    CARBON_DCHECK(is_embedded_value());
    return id_;
  }

  // Converts an ID to an index. Requires that `is_index()` is true.
  //
  // Note `None` is represented as an index ID, and can be converted here.
  constexpr auto AsIndex() const -> int {
    CARBON_DCHECK(is_index());
    return ZeroIndexId - id_;
  }

  // Returns the ID formatted as a lex token payload.
  constexpr auto AsTokenPayload() const -> uint32_t {
    uint32_t payload = id_;
    // Ensure this ID round trips as the token payload.
    CARBON_DCHECK(*this == MakeFromTokenPayload(payload));
    return payload;
  }

  constexpr auto AsRaw() const -> int32_t { return id_; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << Label << "(";
    if (is_embedded_value()) {
      out << "value: " << AsValue();
    } else if (is_index()) {
      out << "index: " << AsIndex();
    } else {
      CARBON_CHECK(!has_value());
      out << "<none>";
    }
    out << ")";
  }

  friend constexpr auto operator==(IntId lhs, IntId rhs) -> bool {
    return lhs.id_ == rhs.id_;
  }
  friend constexpr auto operator<=>(IntId lhs, IntId rhs)
      -> std::strong_ordering {
    return lhs.id_ <=> rhs.id_;
  }

 private:
  friend class IntStore;
  friend Testing::IntStoreTestPeer;

  // The shift needed when adjusting a between a `TokenIdBits`-width integer and
  // a 32-bit integer.
  static constexpr int TokenIdBitsShift = 32 - TokenIdBits;

  // The maximum embedded value in an ID.
  static constexpr int32_t MaxValue =
      std::numeric_limits<int32_t>::max() >> TokenIdBitsShift;

  // The ID value that represents an index of `0`. This is the first ID value
  // representing an index, and all indices are `<=` to this.
  //
  // `ZeroIndexId` is the first index ID, and we encode indices as successive
  // negative numbers counting downwards. The setup allows us to both use a
  // comparison with this ID to distinguish value and index IDs, and to compute
  // the actual index from the ID.
  //
  // The computation of an index in fact is just a subtraction:
  // `ZeroIndexId - id_`. Subtraction is *also* how most CPUs implement the
  // comparison, and so all of this ends up carefully constructed to enable very
  // small code size when testing for an embedded value and when that test fails
  // computing and using the index.
  static constexpr int32_t ZeroIndexId = std::numeric_limits<int32_t>::min() >>
                                         (TokenIdBitsShift + 1);

  // The minimum embedded value in an ID.
  static constexpr int32_t MinValue = ZeroIndexId + 1;

  // The `None` ID, which needs to be placed after the largest index, which
  // count downwards as IDs so below the smallest index ID, in order to optimize
  // the code sequence needed to distinguish between integer and value IDs and
  // to convert index IDs into actual indices small.
  static constexpr int32_t NoneId = std::numeric_limits<int32_t>::min();

  // The `None` index. This is the result of converting a `None` ID into an
  // index. We ensure that conversion can be done so that we can simplify the
  // code that first tries to use an embedded value, then converts to an index
  // and checks that the index is still `None`.
  static const int32_t NoneIndex;

  // Document the specific values of some of these constants to help visualize
  // how the bit patterns map from the above computations.
  //
  // clang-format off: visualizing bit positions
  //
  // Each bit is either `T` for part of the token or `P` as part
  // of the available payload that we use for the ID:
  //
  //                           0bTTTT'TTTT'TPPP'PPPP'PPPP'PPPP'PPPP'PPPP
  static_assert(MaxValue    == 0b0000'0000'0011'1111'1111'1111'1111'1111);
  static_assert(ZeroIndexId == 0b1111'1111'1110'0000'0000'0000'0000'0000);
  static_assert(MinValue    == 0b1111'1111'1110'0000'0000'0000'0000'0001);
  static_assert(NoneId      == 0b1000'0000'0000'0000'0000'0000'0000'0000);
  // clang-format on

  constexpr explicit IntId(int32_t id) : id_(id) {}

  int32_t id_;
};

constexpr IntId IntId::None(IntId::NoneId);

// Note that we initialize the `None` index in a constexpr context which
// ensures there is no UB in forming it. This helps ensure all the ID -> index
// conversions are correct because the `None` ID is at the limit of that range.
constexpr int32_t IntId::NoneIndex = None.AsIndex();

// A canonicalizing value store with deep optimizations for integers.
//
// This stores integers as abstract, signed mathematical integers. The bit width
// of specific `APInt` values, either as inputs or outputs, is disregarded for
// the purpose of canonicalization and the returned integer may use a very
// different bit width `APInt` than was used when adding. There are also
// optimized paths for adding integer values representable using native integer
// types.
//
// Because the integers in the store are canonicalized with only a minimum bit
// width, there are helper functions to coerce them to a specific desired bit
// width for use.
//
// This leverages a significant optimization for small integer values -- rather
// than canonicalizing and making them unique in a `ValueStore`, they are
// directly embedded in the `IntId` itself. Only larger integers are stored in
// an array of `APInt` values and represented as an index in the ID.
class IntStore {
 public:
  // The maximum supported bit width of an integer type.
  // TODO: Pick a maximum size and document it in the design. For now
  // we use 2^^23, because that's the largest size that LLVM supports.
  static constexpr int MaxIntWidth = 1 << 23;

  // Pick a canonical bit width for the provided number of significant bits.
  static auto CanonicalBitWidth(int significant_bits) -> int;

  // Accepts a signed `int64_t` and uses the mathematical signed integer value
  // of it as the added integer value.
  //
  // Returns the ID corresponding to this integer value, storing an `APInt` if
  // necessary to represent it.
  auto Add(int64_t value) -> IntId {
    // First try directly making this into an ID.
    if (IntId id = TryMakeValue(value); id.has_value()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return AddLarge(value);
  }

  // Returns the ID corresponding to this signed integer value, storing an
  // `APInt` if necessary to represent it.
  auto AddSigned(llvm::APInt value) -> IntId {
    // First try directly making this into an ID.
    if (IntId id = TryMakeSignedValue(value); id.has_value()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return AddSignedLarge(std::move(value));
  }

  // Returns the ID corresponding to an equivalent signed integer value for the
  // provided unsigned integer value, storing an `APInt` if necessary to
  // represent it.
  auto AddUnsigned(llvm::APInt value) -> IntId {
    // First try directly making this into an ID.
    if (IntId id = TryMakeUnsignedValue(value); id.has_value()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return AddUnsignedLarge(std::move(value));
  }

  // Returns the value for an ID.
  //
  // This will always be a signed `APInt` with a canonical bit width for the
  // specific integer value in question.
  auto Get(IntId id) const -> llvm::APInt {
    if (id.is_embedded_value()) [[likely]] {
      return llvm::APInt(MinAPWidth, id.AsValue(), /*isSigned=*/true);
    }
    return values_.Get(APIntId(id.AsIndex()));
  }

  // Returns the value for an ID adjusted to a specific bit width.
  //
  // Note that because we store canonical mathematical integers as signed
  // integers, this always sign extends or truncates to the target width. The
  // caller can then use that as a signed or unsigned integer as needed.
  auto GetAtWidth(IntId id, int bit_width) const -> llvm::APInt {
    llvm::APInt value = Get(id);
    if (static_cast<int>(value.getBitWidth()) != bit_width) {
      value = value.sextOrTrunc(bit_width);
    }
    return value;
  }

  // Returns the value for an ID adjusted to the bit width specified with
  // another integer ID.
  //
  // This simply looks up the width integer ID, and then calls the above
  // `GetAtWidth` overload using the value found for it. See that overload for
  // more details.
  auto GetAtWidth(IntId id, IntId bit_width_id) const -> llvm::APInt {
    const llvm::APInt bit_width = Get(bit_width_id);
    CARBON_CHECK(
        bit_width.isStrictlyPositive() && bit_width.isSignedIntN(MinAPWidth),
        "Invalid bit width value: {0}", bit_width);
    return GetAtWidth(id, bit_width.getSExtValue());
  }

  // Accepts a signed `int64_t` and uses the mathematical signed integer value
  // of it as the integer value to lookup. Returns the canonical ID for that
  // value or returns `None` if not in the store.
  auto Lookup(int64_t value) const -> IntId {
    if (IntId id = TryMakeValue(value); id.has_value()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return LookupLarge(value);
  }

  // Looks up the canonical ID for this signed integer value, or returns `None`
  // if not in the store.
  auto LookupSigned(llvm::APInt value) const -> IntId {
    if (IntId id = TryMakeSignedValue(value); id.has_value()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return LookupSignedLarge(std::move(value));
  }

  // Output a YAML description of this data structure. Note that this will only
  // include the integers that required storing, not those successfully embedded
  // into the ID space.
  auto OutputYaml() const -> Yaml::OutputMapping;

  auto array_ref() const -> llvm::ArrayRef<llvm::APInt> {
    return values_.array_ref();
  }
  auto size() const -> size_t { return values_.size(); }

  // Collects the memory usage of the separately stored integers.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

 private:
  friend struct Testing::IntStoreTestPeer;

  // Used for `values_`; tracked using `IntId`'s index range.
  struct APIntId : IdBase<APIntId> {
    static constexpr llvm::StringLiteral Label = "ap_int";
    using ValueType = llvm::APInt;
    static const APIntId None;
    using IdBase::IdBase;
  };

  static constexpr int MinAPWidth = 64;

  static auto MakeIndexOrNone(int index) -> IntId {
    CARBON_DCHECK(index >= 0 && index <= IntId::NoneIndex);
    return IntId(IntId::ZeroIndexId - index);
  }

  // Tries to make a signed 64-bit integer into an embedded value in the ID, and
  // if unable to do that returns the `None` ID.
  static auto TryMakeValue(int64_t value) -> IntId {
    if (IntId::MinValue <= value && value <= IntId::MaxValue) {
      return IntId(value);
    }

    return IntId::None;
  }

  // Tries to make a signed APInt into an embedded value in the ID, and if
  // unable to do that returns the `None` ID.
  static auto TryMakeSignedValue(llvm::APInt value) -> IntId {
    if (value.sge(IntId::MinValue) && value.sle(IntId::MaxValue)) {
      return IntId(value.getSExtValue());
    }

    return IntId::None;
  }

  // Tries to make an unsigned APInt into an embedded value in the ID, and if
  // unable to do that returns the `None` ID.
  static auto TryMakeUnsignedValue(llvm::APInt value) -> IntId {
    if (value.ule(IntId::MaxValue)) {
      return IntId(value.getZExtValue());
    }

    return IntId::None;
  }

  // Canonicalize an incoming signed APInt to the correct bit width.
  static auto CanonicalizeSigned(llvm::APInt value) -> llvm::APInt;

  // Canonicalize an incoming unsigned APInt to the correct bit width.
  static auto CanonicalizeUnsigned(llvm::APInt value) -> llvm::APInt;

  // Helper functions for handling values that are large enough to require an
  // allocated `APInt` for storage. Creating or manipulating that storage is
  // only a few lines of code, but we move these out-of-line because the
  // generated code is big and harms performance for the non-`Large` common
  // case.
  auto AddLarge(int64_t value) -> IntId;
  auto AddSignedLarge(llvm::APInt value) -> IntId;
  auto AddUnsignedLarge(llvm::APInt value) -> IntId;
  auto LookupLarge(int64_t value) const -> IntId;
  auto LookupSignedLarge(llvm::APInt value) const -> IntId;

  // Stores values which don't fit in an IntId. These are always signed.
  CanonicalValueStore<APIntId> values_;
};

constexpr IntStore::APIntId IntStore::APIntId::None(IntId::None.AsIndex());

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_INT_H_
