// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_

#include <type_traits>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/YAMLParser.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// The value of a real literal.
//
// This is either a dyadic fraction (mantissa * 2^exponent) or a decadic
// fraction (mantissa * 10^exponent).
//
// TODO: For SemIR, replace this with a Rational type, per the design:
// docs/design/expressions/literals.md
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

// Corresponds to an integer value represented by an APInt.
struct IntId : public IdBase, public Printable<IntId> {
  using ValueType = const llvm::APInt;
  static const IntId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IdBase::Print(out);
  }
};
constexpr IntId IntId::Invalid(IntId::InvalidIndex);

// Corresponds to a Real value.
struct RealId : public IdBase, public Printable<RealId> {
  using ValueType = const Real;
  static const RealId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "real";
    IdBase::Print(out);
  }
};
constexpr RealId RealId::Invalid(RealId::InvalidIndex);

// Corresponds to a StringRef.
struct StringId : public IdBase, public Printable<StringId> {
  using ValueType = const std::string;
  static const StringId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "str";
    IdBase::Print(out);
  }
};
constexpr StringId StringId::Invalid(StringId::InvalidIndex);

// Adapts StringId for identifiers.
//
// `NameId` relies on the values of this type other than `Invalid` all being
// non-negative.
struct IdentifierId : public IdBase, public Printable<IdentifierId> {
  static const IdentifierId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "strId";
    IdBase::Print(out);
  }
};
constexpr IdentifierId IdentifierId::Invalid(IdentifierId::InvalidIndex);

// Adapts StringId for values of string literals.
struct StringLiteralValueId : public IdBase,
                              public Printable<StringLiteralValueId> {
  static const StringLiteralValueId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "strLit";
    IdBase::Print(out);
  }
};
constexpr StringLiteralValueId StringLiteralValueId::Invalid(
    StringLiteralValueId::InvalidIndex);

namespace Internal {
// Used as a parent class for non-printable types. This is just for
// std::conditional, not as an API.
class ValueStoreNotPrintable {};
}  // namespace Internal

// A simple wrapper for accumulating values, providing IDs to later retrieve the
// value. This does not do deduplication.
//
// IdT::ValueType must represent the type being indexed.
template <typename IdT>
class ValueStore
    : public std::conditional<
          std::is_base_of_v<Printable<typename IdT::ValueType>,
                            typename IdT::ValueType>,
          Yaml::Printable<ValueStore<IdT>>, Internal::ValueStoreNotPrintable> {
 public:
  using ValueType = typename IdT::ValueType;

  // Stores the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdT {
    IdT id = IdT(values_.size());
    CARBON_CHECK(id.index >= 0) << "Id overflow";
    values_.push_back(std::move(value));
    return id;
  }

  // Adds a default constructed value and returns an ID to reference it.
  auto AddDefaultValue() -> IdT {
    auto id = IdT(values_.size());
    values_.resize(id.index + 1);
    return id;
  }

  // Returns a mutable value for an ID.
  auto Get(IdT id) -> ValueType& {
    CARBON_CHECK(id.index >= 0) << id;
    return values_[id.index];
  }

  // Returns the value for an ID.
  auto Get(IdT id) const -> const ValueType& {
    CARBON_CHECK(id.index >= 0) << id;
    return values_[id.index];
  }

  // Reserves space.
  auto Reserve(size_t size) -> void { values_.reserve(size); }

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto i : llvm::seq(values_.size())) {
        auto id = IdT(i);
        map.Add(PrintToString(id), Yaml::OutputScalar(Get(id)));
      }
    });
  }

  auto array_ref() const -> llvm::ArrayRef<ValueType> { return values_; }
  auto size() const -> size_t { return values_.size(); }

 private:
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<std::decay_t<ValueType>, 0> values_;
};

// Storage for StringRefs. The caller is responsible for ensuring storage is
// allocated.
template <>
class ValueStore<StringId> : public Yaml::Printable<ValueStore<StringId>> {
 public:
  // Returns an ID to reference the value. May return an existing ID if the
  // string was previously added.
  auto Add(llvm::StringRef value) -> StringId {
    auto [it, inserted] = map_.insert({value, StringId(values_.size())});
    if (inserted) {
      CARBON_CHECK(it->second.index >= 0) << "Too many unique strings";
      values_.push_back(value);
    }
    return it->second;
  }

  // Returns the value for an ID.
  auto Get(StringId id) const -> llvm::StringRef {
    CARBON_CHECK(id.is_valid());
    return values_[id.index];
  }

  // Returns an ID for the value, or Invalid if not found.
  auto Lookup(llvm::StringRef value) const -> StringId {
    if (auto it = map_.find(value); it != map_.end()) {
      return it->second;
    }
    return StringId::Invalid;
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto [i, val] : llvm::enumerate(values_)) {
        map.Add(PrintToString(StringId(i)), val);
      }
    });
  }

  auto size() const -> size_t { return values_.size(); }

 private:
  llvm::DenseMap<llvm::StringRef, StringId> map_;
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<llvm::StringRef, 0> values_;
};

// A thin wrapper around a `ValueStore<StringId>` that provides a different IdT,
// while using a unified storage for values. This avoids potentially
// duplicative string hash maps, which are expensive.
template <typename IdT>
class StringStoreWrapper : public Printable<StringStoreWrapper<IdT>> {
 public:
  explicit StringStoreWrapper(ValueStore<StringId>* values) : values_(values) {}

  auto Add(llvm::StringRef value) -> IdT {
    return IdT(values_->Add(value).index);
  }

  auto Get(IdT id) const -> llvm::StringRef {
    return values_->Get(StringId(id.index));
  }

  auto Lookup(llvm::StringRef value) const -> IdT {
    return IdT(values_->Lookup(value).index);
  }

  auto Print(llvm::raw_ostream& out) const -> void { out << *values_; }

  auto size() const -> size_t { return values_->size(); }

 private:
  ValueStore<StringId>* values_;
};

// Stores that will be used across compiler phases for a given compilation unit.
// This is provided mainly so that they don't need to be passed separately.
class SharedValueStores : public Yaml::Printable<SharedValueStores> {
 public:
  explicit SharedValueStores()
      : identifiers_(&strings_), string_literal_values_(&strings_) {}

  // Not copyable or movable.
  SharedValueStores(const SharedValueStores&) = delete;
  auto operator=(const SharedValueStores&) -> SharedValueStores& = delete;

  auto identifiers() -> StringStoreWrapper<IdentifierId>& {
    return identifiers_;
  }
  auto identifiers() const -> const StringStoreWrapper<IdentifierId>& {
    return identifiers_;
  }
  auto ints() -> ValueStore<IntId>& { return ints_; }
  auto ints() const -> const ValueStore<IntId>& { return ints_; }
  auto reals() -> ValueStore<RealId>& { return reals_; }
  auto reals() const -> const ValueStore<RealId>& { return reals_; }
  auto string_literal_values() -> StringStoreWrapper<StringLiteralValueId>& {
    return string_literal_values_;
  }
  auto string_literal_values() const
      -> const StringStoreWrapper<StringLiteralValueId>& {
    return string_literal_values_;
  }

  auto OutputYaml(std::optional<llvm::StringRef> filename = std::nullopt) const
      -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&, filename](Yaml::OutputMapping::Map map) {
      if (filename) {
        map.Add("filename", *filename);
      }
      map.Add("shared_values",
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                map.Add("ints", ints_.OutputYaml());
                map.Add("reals", reals_.OutputYaml());
                map.Add("strings", strings_.OutputYaml());
              }));
    });
  }

 private:
  ValueStore<IntId> ints_;
  ValueStore<RealId> reals_;

  ValueStore<StringId> strings_;
  StringStoreWrapper<IdentifierId> identifiers_;
  StringStoreWrapper<StringLiteralValueId> string_literal_values_;
};

}  // namespace Carbon

// Support use of IdentifierId as DenseMap/DenseSet keys.
// TODO: Remove once NameId is used in checking.
template <>
struct llvm::DenseMapInfo<Carbon::IdentifierId>
    : public Carbon::IndexMapInfo<Carbon::IdentifierId> {};
// Support use of StringId as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::StringId>
    : public Carbon::IndexMapInfo<Carbon::StringId> {};

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
