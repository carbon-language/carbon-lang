// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_H_
#define CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_H_

#include "common/hashtable_key_context.h"
#include "common/set.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/value_store_types.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// A wrapper for accumulating immutable values with deduplication, providing IDs
// to later retrieve the value.
//
// `IdT::ValueType` must represent the type being indexed.
//
// `IdT::KeyType` can optionally be present, and if so is used for the argument
// to `Lookup`. It must be valid to use both `KeyType` and `ValueType` as lookup
// types in the underlying `Set`.
template <typename IdT>
class CanonicalValueStore {
 public:
  using ValueType = ValueStoreTypes<IdT>::ValueType;
  using KeyType = ValueStoreTypes<IdT>::KeyType;
  using RefType = ValueStoreTypes<IdT>::RefType;
  using ConstRefType = ValueStoreTypes<IdT>::ConstRefType;

  // Stores a canonical copy of the value and returns an ID to reference it.
  auto Add(ValueType value) -> IdT;

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType { return values_.Get(id); }

  // Looks up the canonical ID for a value, or returns `None` if not in the
  // store.
  auto Lookup(KeyType key) const -> IdT;

  // Reserves space.
  auto Reserve(size_t size) -> void;

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  auto values() const [[clang::lifetimebound]] -> ValueStore<IdT>::Range {
    return values_.values();
  }
  auto size() const -> size_t { return values_.size(); }

  // Collects memory usage of the values and deduplication set.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    auto bytes = set_.ComputeMetrics(KeyContext(&values_)).storage_bytes;
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
  explicit KeyContext(const ValueStore<IdT>* values) : values_(values) {}

  // Note that it is safe to return a `const` reference here as the underlying
  // object's lifetime is provided by the `ValueStore`.
  auto TranslateKey(IdT id) const -> ValueStore<IdT>::ConstRefType {
    return values_->Get(id);
  }

 private:
  const ValueStore<IdT>* values_;
};

template <typename IdT>
auto CanonicalValueStore<IdT>::Add(ValueType value) -> IdT {
  auto make_key = [&] { return IdT(values_.Add(std::move(value))); };
  return set_.Insert(value, make_key, KeyContext(&values_)).key();
}

template <typename IdT>
auto CanonicalValueStore<IdT>::Lookup(KeyType key) const -> IdT {
  if (auto result = set_.Lookup(key, KeyContext(&values_))) {
    return result.key();
  }
  return IdT::None;
}

template <typename IdT>
auto CanonicalValueStore<IdT>::Reserve(size_t size) -> void {
  // Compute the resulting new insert count using the size of values -- the
  // set doesn't have a fast to compute current size.
  if (size > values_.size()) {
    set_.GrowForInsertCount(size - values_.size(), KeyContext(&values_));
  }
  values_.Reserve(size);
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_H_
