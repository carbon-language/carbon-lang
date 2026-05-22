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
// `ValueT` represents the type being stored.
//
// `KeyT` can optionally be different from `ValueT`, and if so is used for the
// argument to `Lookup`. In this case, `ValueT` must provide a `GetAsKey` member
// function that returns the corresponding key.
//
// This template class is designed to be explicitly instantiated to improve
// compile times. Heavy method definitions are in
// `canonical_value_store_impl.h`.
//
// To use a new instantiation:
// 1. Add an `extern template class CanonicalValueStore<...>;` declaration in
//    the header where the instantiation is anchored.
// 2. Add `template class CanonicalValueStore<...>;` definition in the
//    associated `.cpp` file.
// 3. Include `toolchain/base/canonical_value_store_impl.h` in that `.cpp` file.
template <typename IdT, typename KeyT, typename TagIdT = Untagged,
          typename ValueT = KeyT>
class CanonicalValueStore {
 public:
  using IdType = IdT;
  using IdTagType = IdTag<IdT, TagIdT>;
  using KeyType = std::remove_cvref_t<KeyT>;
  using ValueType = ValueStoreTypes<ValueT>::ValueType;
  using RefType = ValueStoreTypes<ValueT>::RefType;
  using ConstRefType = ValueStoreTypes<ValueT>::ConstRefType;

  CanonicalValueStore()
    requires(IdTagIsUntagged<IdTag<IdT, TagIdT>>);
  explicit CanonicalValueStore(IdTagType::TagIdType id,
                               int32_t initial_reserved_ids = 0)
    requires(!IdTagIsUntagged<IdTag<IdT, TagIdT>>);

  ~CanonicalValueStore();
  CanonicalValueStore(CanonicalValueStore&&) noexcept;
  auto operator=(CanonicalValueStore&&) noexcept -> CanonicalValueStore&;

  // Stores a canonical copy of the value and returns an ID to reference it. If
  // the value is already in the store, returns the ID of the existing value.
  auto Add(ValueType value) -> IdT;

  // Returns the value for an ID.
  auto Get(IdT id) const -> ConstRefType { return values_.Get(id); }

  // Looks up the canonical ID for a value, or returns `None` if not in the
  // store.
  auto Lookup(KeyType key) const -> IdT;

  // Reserves space.
  auto Reserve(size_t size) -> void;

  // These are to support printable structures, and are not guaranteed.
  auto OutputYaml() const -> Yaml::OutputMapping;

  auto values() const [[clang::lifetimebound]]
  -> ValueStore<IdT, ValueType, TagIdT>::Range {
    return values_.values();
  }
  auto size() const -> size_t { return values_.size(); }
  auto enumerate() const -> auto { return values_.enumerate(); }

  // Collects memory usage of the values and deduplication set.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  auto GetRawIndex(IdT id) const -> int32_t { return values_.GetRawIndex(id); }

  auto GetIdTag() const -> IdTagType { return values_.GetIdTag(); }

 private:
  class KeyContext;

  static auto GetAsKey(ConstRefType value) -> ConstRefType
    requires std::same_as<KeyT, ValueT>
  {
    return value;
  }

  template <typename T>
  static auto GetAsKey(T&& value) -> decltype(value.GetAsKey()) {
    return value.GetAsKey();
  }

  ValueStore<IdT, ValueType, TagIdT> values_;
  Set<IdT, /*SmallSize=*/0, KeyContext> set_;
};

template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
class CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::KeyContext
    : public TranslatingKeyContext<KeyContext> {
 public:
  explicit KeyContext(const ValueStore<IdT, ValueType, TagIdT>* values)
      : values_(values) {}

  // Note that it is safe to return a reference here as the underlying object's
  // lifetime is provided by the `ValueStore`.
  auto TranslateKey(IdT id) const
      -> decltype(GetAsKey(std::declval<ConstRefType>())) {
    return GetAsKey(values_->Get(id));
  }

 private:
  const ValueStore<IdT, ValueType, TagIdT>* values_;
};

// The definition of `Add` needs to be outside the class body as it relies on
// the definition of `KeyContext` being complete.
//
// Note that the `inline` keyword is necessary here even though this is a
// template because we do explicit instantiations of this template which will
// make inlining this routine impossible despite it being performance sensitive.
template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
inline auto CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::Add(ValueType value)
    -> IdT {
  auto make_key = [&] { return IdT(values_.Add(std::move(value))); };
  return set_.Insert(GetAsKey(value), make_key, KeyContext(&values_)).key();
}

// The definition of `Lookup` needs to be outside the class body as it relies on
// the definition of `KeyContext` being complete.
//
// Note that the `inline` keyword is necessary here even though this is a
// template because we do explicit instantiations of this template which will
// make inlining this routine impossible despite it being performance sensitive.
template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
inline auto CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::Lookup(
    KeyType key) const -> IdT {
  if (auto result = set_.Lookup(key, KeyContext(&values_))) {
    return result.key();
  }
  return IdT::None;
}

// The definition of `Reserve` needs to be outside the class body as it relies
// on the definition of `KeyContext` being complete.
//
// Note that the `inline` keyword is necessary here even though this is a
// template because we do explicit instantiations of this template which will
// make inlining this routine impossible despite it being performance sensitive.
template <typename IdT, typename KeyT, typename TagIdT, typename ValueT>
inline auto CanonicalValueStore<IdT, KeyT, TagIdT, ValueT>::Reserve(size_t size)
    -> void {
  // Compute the resulting new insert count using the size of values -- the
  // set doesn't have a fast to compute current size.
  if (size > values_.size()) {
    set_.GrowForInsertCount(size - values_.size(), KeyContext(&values_));
  }
  values_.Reserve(size);
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_CANONICAL_VALUE_STORE_H_
