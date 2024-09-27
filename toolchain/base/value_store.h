// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_

#include <type_traits>

#include "common/check.h"
#include "common/ostream.h"
#include "common/set.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/YAMLParser.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

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

// Corresponds to an integer value represented by an APInt. This is used both
// for integer literal tokens, which are unsigned and have an unspecified
// bit-width, and integer values in SemIR, which have a signedness and bit-width
// matching their type.
struct IntId : public IdBase, public Printable<IntId> {
  using ValueType = llvm::APInt;
  static const IntId Invalid;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IdBase::Print(out);
  }
};
constexpr IntId IntId::Invalid(IntId::InvalidIndex);

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

  // Typically we want to use `ValueType&` and `const ValueType& to avoid
  // copies, but when the value type is a `StringRef`, we assume external
  // storage for the string data and both our value type and ref type will be
  // `StringRef`. This will preclude mutation of the string data.
  using RefType = std::conditional_t<std::same_as<llvm::StringRef, ValueType>,
                                     llvm::StringRef, ValueType&>;
  using ConstRefType =
      std::conditional_t<std::same_as<llvm::StringRef, ValueType>,
                         llvm::StringRef, const ValueType&>;

  // Stores the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdT {
    IdT id(values_.size());
    CARBON_CHECK(id.index >= 0, "Id overflow");
    values_.push_back(std::move(value));
    return id;
  }

  // Adds a default constructed value and returns an ID to reference it.
  auto AddDefaultValue() -> IdT {
    IdT id(values_.size());
    values_.resize(id.index + 1);
    return id;
  }

  // Returns a mutable value for an ID.
  auto Get(IdT id) -> RefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    return values_[id.index];
  }

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType {
    CARBON_DCHECK(id.index >= 0, "{0}", id);
    return values_[id.index];
  }

  // Reserves space.
  auto Reserve(size_t size) -> void { values_.reserve(size); }

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto i : llvm::seq(values_.size())) {
        IdT id(i);
        map.Add(PrintToString(id), Yaml::OutputScalar(Get(id)));
      }
    });
  }

  // Collects memory usage of the values.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(label.str(), values_);
  }

  auto array_ref() const -> llvm::ArrayRef<ValueType> { return values_; }
  auto size() const -> size_t { return values_.size(); }

 private:
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<std::decay_t<ValueType>, 0> values_;
};

// A wrapper for accumulating immutable values with deduplication, providing IDs
// to later retrieve the value.
//
// IdT::ValueType must represent the type being indexed.
template <typename IdT>
class CanonicalValueStore {
 public:
  using ValueType = typename IdT::ValueType;
  using RefType = typename ValueStore<IdT>::RefType;
  using ConstRefType = typename ValueStore<IdT>::ConstRefType;

  // Stores a canonical copy of the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdT;

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType { return values_.Get(id); }

  // Looks up the canonical ID for a value, or returns invalid if not in the
  // store.
  auto Lookup(ValueType value) const -> IdT;

  // Reserves space.
  auto Reserve(size_t size) -> void;

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  auto array_ref() const -> llvm::ArrayRef<ValueType> {
    return values_.array_ref();
  }
  auto size() const -> size_t { return values_.size(); }

  // Collects memory usage of the values and deduplication set.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    auto bytes =
        set_.ComputeMetrics(KeyContext(values_.array_ref())).storage_bytes;
    mem_usage.Add(MemUsage::ConcatLabel(label, "set_"), bytes, bytes);
  }

 private:
  class KeyContext;

  ValueStore<IdT> values_;
  Set<IdT, /*SmallSize=*/0, KeyContext> set_;
};

template <typename IdT>
class CanonicalValueStore<IdT>::KeyContext
    : public TranslatingKeyContext<KeyContext> {
 public:
  explicit KeyContext(llvm::ArrayRef<ValueType> values) : values_(values) {}

  // Note that it is safe to return a `const` reference here as the underlying
  // object's lifetime is provided by the `store_`.
  auto TranslateKey(IdT id) const -> const ValueType& {
    return values_[id.index];
  }

 private:
  llvm::ArrayRef<ValueType> values_;
};

template <typename IdT>
auto CanonicalValueStore<IdT>::Add(ValueType value) -> IdT {
  auto make_key = [&] { return IdT(values_.Add(std::move(value))); };
  return set_.Insert(value, make_key, KeyContext(values_.array_ref())).key();
}

template <typename IdT>
auto CanonicalValueStore<IdT>::Lookup(ValueType value) const -> IdT {
  if (auto result = set_.Lookup(value, KeyContext(values_.array_ref()))) {
    return result.key();
  }
  return IdT::Invalid;
}

template <typename IdT>
auto CanonicalValueStore<IdT>::Reserve(size_t size) -> void {
  // Compute the resulting new insert count using the size of values -- the
  // set doesn't have a fast to compute current size.
  if (size > values_.size()) {
    set_.GrowForInsertCount(size - values_.size(),
                            KeyContext(values_.array_ref()));
  }
  values_.Reserve(size);
}

using FloatValueStore = CanonicalValueStore<FloatId>;

// Stores that will be used across compiler phases for a given compilation unit.
// This is provided mainly so that they don't need to be passed separately.
class SharedValueStores : public Yaml::Printable<SharedValueStores> {
 public:
  explicit SharedValueStores() = default;

  // Not copyable or movable.
  SharedValueStores(const SharedValueStores&) = delete;
  auto operator=(const SharedValueStores&) -> SharedValueStores& = delete;

  auto identifiers() -> CanonicalValueStore<IdentifierId>& {
    return identifiers_;
  }
  auto identifiers() const -> const CanonicalValueStore<IdentifierId>& {
    return identifiers_;
  }
  auto ints() -> CanonicalValueStore<IntId>& { return ints_; }
  auto ints() const -> const CanonicalValueStore<IntId>& { return ints_; }
  auto reals() -> ValueStore<RealId>& { return reals_; }
  auto reals() const -> const ValueStore<RealId>& { return reals_; }
  auto floats() -> FloatValueStore& { return floats_; }
  auto floats() const -> const FloatValueStore& { return floats_; }
  auto string_literal_values() -> CanonicalValueStore<StringLiteralValueId>& {
    return string_literals_;
  }
  auto string_literal_values() const
      -> const CanonicalValueStore<StringLiteralValueId>& {
    return string_literals_;
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
                map.Add("identifiers", identifiers_.OutputYaml());
                map.Add("strings", string_literals_.OutputYaml());
              }));
    });
  }

  // Collects memory usage for the various shared stores.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "ints_"), ints_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "reals_"), reals_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "floats_"), floats_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "identifiers_"),
                      identifiers_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "string_literals_"),
                      string_literals_);
  }

 private:
  CanonicalValueStore<IntId> ints_;
  ValueStore<RealId> reals_;
  FloatValueStore floats_;

  CanonicalValueStore<IdentifierId> identifiers_;
  CanonicalValueStore<StringLiteralValueId> string_literals_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_VALUE_STORE_H_
