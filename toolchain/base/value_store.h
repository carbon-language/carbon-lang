// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/base/index_base.h"

namespace Carbon {

// Corresponds to a StringRef.
struct StringId : public IndexBase, public Printable<StringId> {
  static const StringId Invalid;
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "str";
    IndexBase::Print(out);
  }
};
constexpr StringId StringId::Invalid(StringId::InvalidIndex);

// Corresponds to an integer value represented by an APInt.
struct IntegerId : public IndexBase, public Printable<IntegerId> {
  static const IntegerId Invalid;
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IndexBase::Print(out);
  }
};
constexpr IntegerId IntegerId::Invalid(IntegerId::InvalidIndex);

// Corresponds to a RealValue.
struct RealId : public IndexBase, public Printable<RealId> {
  static const RealId Invalid;
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "real";
    IndexBase::Print(out);
  }
};
constexpr RealId RealId::Invalid(RealId::InvalidIndex);

// A simple wrapper for accumulating values, providing IDs to later retrieve the
// value. This does not do deduplication.
template <typename ValueT, typename IdT>
class ValueStore {
 public:
  // Stores the value and returns an ID to reference it.
  auto Add(ValueT value) -> IdT {
    auto id = IdT(values_.size());
    values_.push_back(std::move(value));
    return id;
  }

  // Returns the value for an ID.
  auto Get(IdT id) const -> const ValueT& {
    CARBON_CHECK(id.is_valid());
    return values_[id.index];
  }

 private:
  llvm::SmallVector<ValueT> values_;
};

// Storage for StringRefs. The caller is responsible for ensuring storage is
// allocated.
class StringStore {
 public:
  // Returns an ID to reference the value. May return an existing ID if the
  // string was previously added.
  auto Add(llvm::StringRef value) -> StringId {
    auto [it, inserted] = map_.insert({value, StringId(values_.size())});
    if (inserted) {
      values_.push_back(value);
    }
    return it->second;
  }

  // Returns the value for an ID.
  auto Get(StringId id) const -> llvm::StringRef {
    CARBON_CHECK(id.is_valid());
    return values_[id.index];
  }

 private:
  llvm::StringMap<StringId> map_;
  llvm::SmallVector<llvm::StringRef> values_;
};

// The value of a real literal.
//
// This is either a dyadic fraction (mantissa * 2^exponent) or a decadic
// fraction (mantissa * 10^exponent).
//
// `RealLiteralValue` carries a reference back to `TokenizedBuffer` which can be
// invalidated if the buffer is edited or destroyed.
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
  bool is_decimal;
};

// Stores that will be used across compiler steps. This is provided mainly so
// that they don't need to be passed separately.
class CompileValueStores {
 public:
  auto integers() -> ValueStore<llvm::APInt, IntegerId>& { return integers_; }
  auto reals() -> ValueStore<Real, RealId>& { return reals_; }
  auto strings() -> StringStore& { return strings_; }

 private:
  ValueStore<llvm::APInt, IntegerId> integers_;
  ValueStore<Real, RealId> reals_;
  StringStore strings_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
