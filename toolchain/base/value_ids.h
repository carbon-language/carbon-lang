// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_IDS_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_IDS_H_

#include <limits>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/YAMLParser.h"
#include "toolchain/base/index_base.h"

namespace Carbon {

// Valid IDs which are associated with tokens during lexing need to fit into a
// compressed storage space, which may influence the specific formulation of the
// ID. Note that there may still be IDs either not associated with tokens or
// computed after lexing outside of this range.
constexpr int TokenIdBits = 23;

// The value of a real literal token.
//
// This is either a dyadic fraction (mantissa * 2^exponent) or a decadic
// fraction (mantissa * 10^exponent).
//
// These values are not canonicalized, because we don't expect them to repeat
// and don't use them in SemIR values.
class Real : public Printable<Real> {
 public:
  auto Print(llvm::raw_ostream& output_stream) const -> void {
    mantissa.print(output_stream, /*isSigned=*/false);
    output_stream << "*" << (is_decimal ? "10" : "2") << "^" << exponent;
  }

  // The mantissa, represented as an unsigned integer.
  llvm::APInt mantissa;

  // The exponent, represented as a signed integer.
  llvm::APInt exponent;

  // If false, the value is mantissa * 2^exponent.
  // If true, the value is mantissa * 10^exponent.
  // TODO: This field increases Real from 32 bytes to 40 bytes. Consider
  // changing how it's tracked for space savings.
  bool is_decimal;
};

// Corresponds to a canonicalized integer value. This is used both for integer
// literal tokens, and integer values in SemIR. These always represent the
// abstract mathematical value -- signed and regardless of the needed precision.
//
// Small values are internalized into the ID itself. Large values are
// represented as an index into an array of `APInt`s with a canonicalized bit
// width.
class IntId : public Printable<IntId> {
 public:
  using ValueType = llvm::APInt;

  static const IntId Invalid;

  static auto MakeIndexOrInvalid(int index) -> IntId {
    CARBON_DCHECK(index >= 0 && index <= InvalidIndex);
    return IntId(ZeroIndexId - index);
  }

  static auto MakeFromTokenPayload(uint32_t payload) -> IntId {
    // Token-associated IDs are signed `TokenIdBits` integers, so force the sign
    // extension from that bit.
    constexpr int Shift = 32 - TokenIdBits;
    return IntId(static_cast<int32_t>(payload << Shift) >> Shift);
  }

  // Tries to make a signed 64-bit integer into an embedded value in the ID, and
  // if unable to do that returns the `Invalid` ID.
  static auto TryMakeValue(int64_t value) -> IntId {
    if (MinValue <= value && value <= MaxValue) {
      return IntId(value);
    }

    return Invalid;
  }

  // Tries to make a signed APInt into an embedded value in the ID, and if
  // unable to do that returns the `Invalid` ID.
  static auto TryMakeSignedValue(llvm::APInt value) -> IntId {
    if (value.sge(MinValue) && value.sle(MaxValue)) {
      return IntId(value.getSExtValue());
    }

    return Invalid;
  }

  // Tries to make an unsigned APInt into an embedded value in the ID, and if
  // unable to do that returns the `Invalid` ID.
  static auto TryMakeUnsignedValue(llvm::APInt value) -> IntId {
    if (value.ule(MaxValue)) {
      return IntId(value.getZExtValue());
    }

    return Invalid;
  }

  // Construct an ID from a raw 32-bit ID value.
  static auto MakeRaw(int32_t raw_id) -> IntId { return IntId(raw_id); }

  constexpr auto is_valid() const -> bool { return id_ != InvalidId; }
  constexpr auto is_value() const -> bool { return id_ > ZeroIndexId; }
  constexpr auto is_index() const -> bool { return id_ <= ZeroIndexId; }

  auto AsValue() const -> int {
    CARBON_DCHECK(is_value());
    return id_;
  }

  constexpr auto AsIndex() const -> int {
    CARBON_DCHECK(is_index());
    return ZeroIndexId - id_;
  }

  constexpr auto AsRaw() const -> int32_t { return id_; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int [";
    if (is_value()) {
      out << "value: " << AsValue() << "]";
    } else if (is_index()) {
      out << "index: " << AsIndex() << "]";
    } else {
      CARBON_CHECK(!is_valid());
      out << "invalid]";
    }
  }

  friend constexpr auto operator==(IntId lhs, IntId rhs) -> bool {
    return lhs.id_ == rhs.id_;
  }
  friend constexpr auto operator<=>(IntId lhs, IntId rhs)
      -> std::strong_ordering {
    return lhs.id_ <=> rhs.id_;
  }

 private:
  // We need all the values from maximum to minimum and a healthy range of
  // indices to all fit within the token ID bits.
  //
  // We represent this as a signed TokenIdBits-bit 2s compliment integer. The
  // sign extension from TokenIdBits to a register size can be folded into the
  // shift used to extract from compressed bitfield storage.
  //
  // We then divide the smallest 1/4th of the space to indices, and the larger
  // 3/4ths to embedded values. For 23-bits total this still gives us 2 million
  // unique integers larger than the embedded ones, which would be difficult to
  // fill without exceeding the number of tokens we can lex (8 million). For
  // non-token based integers, the indics can continue downward to the 32-bit
  // signed integer minimum.
  //
  // Note that the invalid ID can't be used with a token. This is OK as we
  // expect invalid tokens to be *error* tokens and not need to represent an
  // invalid integer.
  static constexpr int TokenIdBitsShift = 32 - TokenIdBits;
  static constexpr int32_t MaxValue =
      std::numeric_limits<int32_t>::max() >> TokenIdBitsShift;
  static constexpr int32_t ZeroIndexId = std::numeric_limits<int32_t>::min() >>
                                         (TokenIdBitsShift + 1);
  static constexpr int32_t MinValue = ZeroIndexId + 1;
  static constexpr int32_t InvalidId = std::numeric_limits<int32_t>::min();
  static constexpr int32_t InvalidIndex = ZeroIndexId - InvalidId;

  // Document the specific values of these constants to help visualize how the
  // bit patterns map from the above computations.
  //
  // Each bit is either `T` for part of the token or `P` as part
  // of the available payload that we use for the ID:
  //
  // clang-format off: visualizing bit positions
  //
  //                           0bTTTT'TTTT'TPPP'PPPP'PPPP'PPPP'PPPP'PPPP
  static_assert(MaxValue    == 0b0000'0000'0011'1111'1111'1111'1111'1111);
  static_assert(ZeroIndexId == 0b1111'1111'1110'0000'0000'0000'0000'0000);
  static_assert(MinValue    == 0b1111'1111'1110'0000'0000'0000'0000'0001);
  static_assert(InvalidId   == 0b1000'0000'0000'0000'0000'0000'0000'0000);
  // clang-format on

  constexpr explicit IntId(int32_t id) : id_(id) {}

  int32_t id_;
};
constexpr IntId IntId::Invalid(IntId::InvalidId);

// Corresponds to a float value represented by an APFloat. This is used for
// floating-point values in SemIR.
struct FloatId : public IdBase, public Printable<FloatId> {
  using ValueType = llvm::APFloat;
  static const FloatId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "float";
    IdBase::Print(out);
  }
};
constexpr FloatId FloatId::Invalid(FloatId::InvalidIndex);

// Corresponds to a Real value.
struct RealId : public IdBase, public Printable<RealId> {
  using ValueType = Real;
  static const RealId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "real";
    IdBase::Print(out);
  }
};
constexpr RealId RealId::Invalid(RealId::InvalidIndex);

// Corresponds to StringRefs for identifiers.
//
// `NameId` relies on the values of this type other than `Invalid` all being
// non-negative.
struct IdentifierId : public IdBase, public Printable<IdentifierId> {
  using ValueType = llvm::StringRef;
  static const IdentifierId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "identifier";
    IdBase::Print(out);
  }
};
constexpr IdentifierId IdentifierId::Invalid(IdentifierId::InvalidIndex);

// Corresponds to StringRefs for string literals.
struct StringLiteralValueId : public IdBase,
                              public Printable<StringLiteralValueId> {
  using ValueType = llvm::StringRef;
  static const StringLiteralValueId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "string";
    IdBase::Print(out);
  }
};
constexpr StringLiteralValueId StringLiteralValueId::Invalid(
    StringLiteralValueId::InvalidIndex);

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_IDS_H_
