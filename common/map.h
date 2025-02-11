// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

#include <algorithm>
#include <concepts>
#include <utility>

#include "common/check.h"
#include "common/hashtable_key_context.h"
#include "common/raw_hashtable.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// Forward declarations to resolve cyclic references.
template <typename KeyT, typename ValueT, typename KeyContextT>
class MapView;
template <typename KeyT, typename ValueT, typename KeyContextT>
class MapBase;
template <typename KeyT, typename ValueT, ssize_t SmallSize,
          typename KeyContextT>
class Map;

// A read-only view type for a map from key to value.
//
// This view is a cheap-to-copy type that should be passed by value, but
// provides view or read-only reference semantics to the underlying map data
// structure.
//
// This should always be preferred to a `const`-ref parameter for the `MapBase`
// or `Map` type as it provides more flexibility and a cleaner API.
//
// Note that while this type is a read-only view, that applies to the underlying
// *map* data structure, not the individual entries stored within it. Those can
// be mutated freely as long as both the hashes and equality of the keys are
// preserved. If we applied a deep-`const` design here, it would prevent using
// this type in many useful situations where the elements are mutated but the
// associative container is not. A view of immutable data can always be obtained
// by using `MapView<const T, const V>`, and we enable conversions to more-const
// views. This mirrors the semantics of views like `std::span`.
//
// A specific `KeyContextT` type can optionally be provided to configure how
// keys will be hashed and compared. The default is `DefaultKeyContext` which is
// stateless and will hash using `Carbon::HashValue` and compare using
// `operator==`. Every method accepting a lookup key or operating on the keys in
// the table will also accept an instance of this type. For stateless context
// types, including the default, an instance will be default constructed if not
// provided to these methods. However, stateful contexts should be constructed
// and passed in explicitly. The context type should be small and reasonable to
// pass by value, often a wrapper or pointer to the relevant context needed for
// hashing and comparing keys. For more details about the key context, see
// `hashtable_key_context.h`.
template <typename InputKeyT, typename InputValueT,
          typename InputKeyContextT = DefaultKeyContext>
class MapView
    : RawHashtable::ViewImpl<InputKeyT, InputValueT, InputKeyContextT> {
  using ImplT =
      RawHashtable::ViewImpl<InputKeyT, InputValueT, InputKeyContextT>;
  using EntryT = typename ImplT::EntryT;

 public:
  using KeyT = typename ImplT::KeyT;
  using ValueT = typename ImplT::ValueT;
  using KeyContextT = typename ImplT::KeyContextT;
  using MetricsT = typename ImplT::MetricsT;

  // This type represents the result of lookup operations. It encodes whether
  // the lookup was a success as well as accessors for the key and value.
  class LookupKVResult {
   public:
    LookupKVResult() = default;
    explicit LookupKVResult(EntryT* entry) : entry_(entry) {}

    explicit operator bool() const { return entry_ != nullptr; }

    auto key() const -> KeyT& { return entry_->key(); }
    auto value() const -> ValueT& { return entry_->value(); }

   private:
    EntryT* entry_ = nullptr;
  };

  // Enable implicit conversions that add `const`-ness to either key or value
  // type. This is always safe to do with a view. We use a template to avoid
  // needing all 3 versions.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  MapView(MapView<OtherKeyT, OtherValueT, KeyContextT> other_view)
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<KeyT, const OtherKeyT>) &&
            (std::same_as<ValueT, OtherValueT> ||
             std::same_as<ValueT, const OtherValueT>)
      : ImplT(other_view) {}

  // Tests whether a key is present in the map.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key,
                KeyContextT key_context = KeyContextT()) const -> bool;

  // Lookup a key in the map.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key,
              KeyContextT key_context = KeyContextT()) const -> LookupKVResult;

  // Lookup a key in the map and try to return a pointer to its value. Returns
  // null on a missing key.
  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const
      -> ValueT* requires(std::default_initializable<KeyContextT>);

  // Run the provided callback for every key and value in the map.
  template <typename CallbackT>
  auto ForEach(CallbackT callback) -> void
    requires(std::invocable<CallbackT, KeyT&, ValueT&>);

  // This routine is relatively inefficient and only intended for use in
  // benchmarking or logging of performance anomalies. The specific metrics
  // returned have no specific guarantees beyond being informative in
  // benchmarks.
  auto ComputeMetrics(KeyContextT key_context = KeyContextT()) -> MetricsT {
    return ImplT::ComputeMetricsImpl(key_context);
  }

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize,
            typename KeyContextT>
  friend class Map;
  friend class MapBase<KeyT, ValueT, KeyContextT>;
  friend class MapView<const KeyT, ValueT, KeyContextT>;
  friend class MapView<KeyT, const ValueT, KeyContextT>;
  friend class MapView<const KeyT, const ValueT, KeyContextT>;

  MapView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  MapView(ImplT base) : ImplT(base) {}
  MapView(ssize_t size, RawHashtable::Storage* storage)
      : ImplT(size, storage) {}
};

// A base class for a `Map` type that remains mutable while type-erasing the
// `SmallSize` (SSO) template parameter.
//
// A pointer or reference to this type is the preferred way to pass a mutable
// handle to a `Map` type across API boundaries as it avoids encoding specific
// SSO sizing information while providing a near-complete mutable API.
template <typename InputKeyT, typename InputValueT,
          typename InputKeyContextT = DefaultKeyContext>
class MapBase : protected RawHashtable::BaseImpl<InputKeyT, InputValueT,
                                                 InputKeyContextT> {
 protected:
  using ImplT =
      RawHashtable::BaseImpl<InputKeyT, InputValueT, InputKeyContextT>;
  using EntryT = typename ImplT::EntryT;

 public:
  using KeyT = typename ImplT::KeyT;
  using ValueT = typename ImplT::ValueT;
  using KeyContextT = typename ImplT::KeyContextT;
  using ViewT = MapView<KeyT, ValueT, KeyContextT>;
  using LookupKVResult = typename ViewT::LookupKVResult;
  using MetricsT = typename ImplT::MetricsT;

  // The result type for insertion operations both indicates whether an insert
  // was needed (as opposed to finding an existing element), and provides access
  // to the element's key and value.
  class InsertKVResult {
   public:
    InsertKVResult() = default;
    explicit InsertKVResult(bool inserted, EntryT& entry)
        : entry_(&entry), inserted_(inserted) {}

    auto is_inserted() const -> bool { return inserted_; }

    auto key() const -> KeyT& { return entry_->key(); }
    auto value() const -> ValueT& { return entry_->value(); }

   private:
    EntryT* entry_;
    bool inserted_;
  };

  // Implicitly convertible to the relevant view type.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->view_impl(); }

  // We can't chain the above conversion with the conversions on `ViewT` to add
  // const, so explicitly support adding const to produce a view here.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator MapView<OtherKeyT, OtherValueT, KeyContextT>() const
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<const KeyT, OtherKeyT>) &&
            (std::same_as<ValueT, OtherValueT> ||
             std::same_as<const ValueT, OtherValueT>)
  {
    return ViewT(*this);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key,
                KeyContextT key_context = KeyContextT()) const -> bool {
    return ViewT(*this).Contains(lookup_key, key_context);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key,
              KeyContextT key_context = KeyContextT()) const -> LookupKVResult {
    return ViewT(*this).Lookup(lookup_key, key_context);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const
      -> ValueT* requires(std::default_initializable<KeyContextT>) {
        return ViewT(*this)[lookup_key];
      }

  // Convenience forwarder to the view type.
  template <typename CallbackT>
  auto ForEach(CallbackT callback) const -> void
    requires(std::invocable<CallbackT, KeyT&, ValueT&>)
  {
    return ViewT(*this).ForEach(callback);
  }

  // Convenience forwarder to the view type.
  auto ComputeMetrics(KeyContextT key_context = KeyContextT()) const
      -> MetricsT {
    return ViewT(*this).ComputeMetrics(key_context);
  }

  // Insert a key and value into the map. If the key is already present, the new
  // value is discarded and the existing value preserved.
  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key, ValueT new_v,
              KeyContextT key_context = KeyContextT()) -> InsertKVResult;

  // Insert a key into the map and call the provided callback if necessary to
  // produce a new value when no existing value is found.
  //
  // Example: `m.Insert(key, [] { return default_value; });`
  //
  // TODO: The `;` formatting below appears to be bugs in clang-format with
  // concepts that should be filed upstream.
  template <typename LookupKeyT, typename ValueCallbackT>
  auto Insert(LookupKeyT lookup_key, ValueCallbackT value_cb,
              KeyContextT key_context = KeyContextT()) -> InsertKVResult
    requires(
        !std::same_as<ValueT, ValueCallbackT> &&
        std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
  ;

  // Lookup a key in the map and if missing insert it and call the provided
  // callback to in-place construct both the key and value. The lookup key is
  // passed through to the callback so it needn't be captured and can be kept in
  // a register argument throughout.
  //
  // Example:
  // ```cpp
  //   m.Insert("widget", [](MyStringViewType lookup_key, void* key_storage,
  //                         void* value_storage) {
  //     new (key_storage) MyStringType(lookup_key);
  //     new (value_storage) MyValueType(....);
  //   });
  // ```
  template <typename LookupKeyT, typename InsertCallbackT>
  auto Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb,
              KeyContextT key_context = KeyContextT()) -> InsertKVResult
    requires(!std::same_as<ValueT, InsertCallbackT> &&
             std::invocable<InsertCallbackT, LookupKeyT, void*, void*>);

  // Replace a key's value in a map if already present or insert it if not
  // already present. The new value is always used.
  template <typename LookupKeyT>
  auto Update(LookupKeyT lookup_key, ValueT new_v,
              KeyContextT key_context = KeyContextT()) -> InsertKVResult;

  // Lookup or insert a key into the map, and set it's value to the result of
  // the `value_cb` callback. The callback is always run and its result is
  // always used, whether the key was already in the map or not. Any existing
  // value is replaced with the result.
  //
  // Example: `m.Update(key, [] { return new_value; });`
  template <typename LookupKeyT, typename ValueCallbackT>
  auto Update(LookupKeyT lookup_key, ValueCallbackT value_cb,
              KeyContextT key_context = KeyContextT()) -> InsertKVResult
    requires(
        !std::same_as<ValueT, ValueCallbackT> &&
        std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
  ;

  // Lookup or insert a key into the map. If not already present and the key is
  // inserted, the `insert_cb` is used to construct the new key and value in
  // place. When inserting, the lookup key is passed through to the callback so
  // it needn't be captured and can be kept in a register argument throughout.
  // If the key was already present, the `update_cb` is called to update the
  // existing key and value as desired.
  //
  // Example of counting occurrences:
  // ```cpp
  //   m.Update(item, /*insert_cb=*/[](MyStringViewType lookup_key,
  //                                   void* key_storage, void* value_storage) {
  //                    new (key_storage) MyItem(lookup_key);
  //                    new (value_storage) Count(1);
  //                  },
  //                  /*update_cb=*/[](MyItem& /*key*/, Count& count) {
  //                    ++count;
  //                  });
  // ```
  template <typename LookupKeyT, typename InsertCallbackT,
            typename UpdateCallbackT>
  auto Update(LookupKeyT lookup_key, InsertCallbackT insert_cb,
              UpdateCallbackT update_cb,
              KeyContextT key_context = KeyContextT()) -> InsertKVResult
    requires(!std::same_as<ValueT, InsertCallbackT> &&
             std::invocable<InsertCallbackT, LookupKeyT, void*, void*> &&
             std::invocable<UpdateCallbackT, KeyT&, ValueT&>);

  // Grow the map to a specific allocation size.
  //
  // This will grow the map's hashtable if necessary for it to have an
  // allocation size of `target_alloc_size` which must be a power of two. Note
  // that this will not allow that many keys to be inserted, but a smaller
  // number based on the maximum load factor. If a specific number of insertions
  // need to be achieved without triggering growth, use the `GrowForInsertCount`
  // method.
  auto GrowToAllocSize(ssize_t target_alloc_size,
                       KeyContextT key_context = KeyContextT()) -> void;

  // Grow the map sufficiently to allow inserting the specified number of keys.
  auto GrowForInsertCount(ssize_t count,
                          KeyContextT key_context = KeyContextT()) -> void;

  // Erase a key from the map.
  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key, KeyContextT key_context = KeyContextT())
      -> bool;

  // Clear all key/value pairs from the map but leave the underlying hashtable
  // allocated and in place.
  auto Clear() -> void;

 protected:
  using ImplT::ImplT;
};

// A data structure mapping from key to value.
//
// This map also supports small size optimization (or "SSO"). The provided
// `SmallSize` type parameter indicates the size of an embedded buffer for
// storing maps small enough to fit. The default is zero, which always allocates
// a heap buffer on construction. When non-zero, must be a multiple of the
// `MaxGroupSize` which is currently 16. The library will check that the size is
// valid and provide an error at compile time if not. We don't automatically
// select the next multiple or otherwise fit the size to the constraints to make
// it clear in the code how much memory is used by the SSO buffer.
//
// This data structure optimizes heavily for small key types that are cheap to
// move and even copy. Using types with large keys or expensive to copy keys may
// create surprising performance bottlenecks. A `std::string` key should be fine
// with generally small strings, but if some or many strings are large heap
// allocations the performance of hashtable routines may be unacceptably bad and
// another data structure or key design is likely preferable.
//
// Note that this type should typically not appear on API boundaries; either
// `MapBase` or `MapView` should be used instead.
template <typename InputKeyT, typename InputValueT, ssize_t SmallSize = 0,
          typename InputKeyContextT = DefaultKeyContext>
class Map : public RawHashtable::TableImpl<
                MapBase<InputKeyT, InputValueT, InputKeyContextT>, SmallSize> {
  using BaseT = MapBase<InputKeyT, InputValueT, InputKeyContextT>;
  using ImplT = RawHashtable::TableImpl<BaseT, SmallSize>;

 public:
  using KeyT = typename BaseT::KeyT;
  using ValueT = typename BaseT::ValueT;

  Map() = default;
  Map(const Map& arg) = default;
  Map(Map&& arg) noexcept = default;
  auto operator=(const Map& arg) -> Map& = default;
  auto operator=(Map&& arg) noexcept -> Map& = default;

  // Reset the entire state of the hashtable to as it was when constructed,
  // throwing away any intervening allocations.
  auto Reset() -> void;
};

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto MapView<InputKeyT, InputValueT, InputKeyContextT>::Contains(
    LookupKeyT lookup_key, KeyContextT key_context) const -> bool {
  return this->LookupEntry(lookup_key, key_context) != nullptr;
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto MapView<InputKeyT, InputValueT, InputKeyContextT>::Lookup(
    LookupKeyT lookup_key, KeyContextT key_context) const -> LookupKVResult {
  return LookupKVResult(this->LookupEntry(lookup_key, key_context));
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto MapView<InputKeyT, InputValueT, InputKeyContextT>::operator[](
    LookupKeyT lookup_key) const
    -> ValueT* requires(std::default_initializable<KeyContextT>) {
      auto result = Lookup(lookup_key, KeyContextT());
      return result ? &result.value() : nullptr;
    }

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename CallbackT>
auto MapView<InputKeyT, InputValueT, InputKeyContextT>::ForEach(
    CallbackT callback) -> void
  requires(std::invocable<CallbackT, KeyT&, ValueT&>)
{
  this->ForEachEntry(
      [callback](EntryT& entry) { callback(entry.key(), entry.value()); },
      [](auto...) {});
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
[[clang::always_inline]] auto
MapBase<InputKeyT, InputValueT, InputKeyContextT>::Insert(
    LookupKeyT lookup_key, ValueT new_v, KeyContextT key_context)
    -> InsertKVResult {
  return Insert(
      lookup_key,
      [&new_v](LookupKeyT lookup_key, void* key_storage, void* value_storage) {
        new (key_storage) KeyT(lookup_key);
        new (value_storage) ValueT(std::move(new_v));
      },
      key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT, typename ValueCallbackT>
[[clang::always_inline]] auto
MapBase<InputKeyT, InputValueT, InputKeyContextT>::Insert(
    LookupKeyT lookup_key, ValueCallbackT value_cb, KeyContextT key_context)
    -> InsertKVResult
  requires(
      !std::same_as<ValueT, ValueCallbackT> &&
      std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
{
  return Insert(
      lookup_key,
      [&value_cb](LookupKeyT lookup_key, void* key_storage,
                  void* value_storage) {
        new (key_storage) KeyT(lookup_key);
        new (value_storage) ValueT(value_cb());
      },
      key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT, typename InsertCallbackT>
[[clang::always_inline]] auto
MapBase<InputKeyT, InputValueT, InputKeyContextT>::Insert(
    LookupKeyT lookup_key, InsertCallbackT insert_cb, KeyContextT key_context)
    -> InsertKVResult
  requires(!std::same_as<ValueT, InsertCallbackT> &&
           std::invocable<InsertCallbackT, LookupKeyT, void*, void*>)
{
  auto [entry, inserted] = this->InsertImpl(lookup_key, key_context);
  CARBON_DCHECK(entry, "Should always result in a valid index.");

  if (LLVM_LIKELY(!inserted)) {
    return InsertKVResult(false, *entry);
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage),
            static_cast<void*>(&entry->value_storage));
  return InsertKVResult(true, *entry);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
[[clang::always_inline]] auto
MapBase<InputKeyT, InputValueT, InputKeyContextT>::Update(
    LookupKeyT lookup_key, ValueT new_v, KeyContextT key_context)
    -> InsertKVResult {
  return Update(
      lookup_key,
      [&new_v](LookupKeyT lookup_key, void* key_storage, void* value_storage) {
        new (key_storage) KeyT(lookup_key);
        new (value_storage) ValueT(std::move(new_v));
      },
      [&new_v](KeyT& /*key*/, ValueT& value) {
        value.~ValueT();
        new (&value) ValueT(std::move(new_v));
      },
      key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT, typename ValueCallbackT>
[[clang::always_inline]] auto
MapBase<InputKeyT, InputValueT, InputKeyContextT>::Update(
    LookupKeyT lookup_key, ValueCallbackT value_cb, KeyContextT key_context)
    -> InsertKVResult
  requires(
      !std::same_as<ValueT, ValueCallbackT> &&
      std::convertible_to<decltype(std::declval<ValueCallbackT>()()), ValueT>)
{
  return Update(
      lookup_key,
      [&value_cb](LookupKeyT lookup_key, void* key_storage,
                  void* value_storage) {
        new (key_storage) KeyT(lookup_key);
        new (value_storage) ValueT(value_cb());
      },
      [&value_cb](KeyT& /*key*/, ValueT& value) {
        value.~ValueT();
        new (&value) ValueT(value_cb());
      },
      key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT, typename InsertCallbackT,
          typename UpdateCallbackT>
[[clang::always_inline]] auto
MapBase<InputKeyT, InputValueT, InputKeyContextT>::Update(
    LookupKeyT lookup_key, InsertCallbackT insert_cb, UpdateCallbackT update_cb,
    KeyContextT key_context) -> InsertKVResult
  requires(!std::same_as<ValueT, InsertCallbackT> &&
           std::invocable<InsertCallbackT, LookupKeyT, void*, void*> &&
           std::invocable<UpdateCallbackT, KeyT&, ValueT&>)
{
  auto [entry, inserted] = this->InsertImpl(lookup_key, key_context);
  CARBON_DCHECK(entry, "Should always result in a valid index.");

  if (LLVM_LIKELY(!inserted)) {
    update_cb(entry->key(), entry->value());
    return InsertKVResult(false, *entry);
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage),
            static_cast<void*>(&entry->value_storage));
  return InsertKVResult(true, *entry);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto MapBase<InputKeyT, InputValueT, InputKeyContextT>::GrowToAllocSize(
    ssize_t target_alloc_size, KeyContextT key_context) -> void {
  this->GrowToAllocSizeImpl(target_alloc_size, key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto MapBase<InputKeyT, InputValueT, InputKeyContextT>::GrowForInsertCount(
    ssize_t count, KeyContextT key_context) -> void {
  this->GrowForInsertCountImpl(count, key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto MapBase<InputKeyT, InputValueT, InputKeyContextT>::Erase(
    LookupKeyT lookup_key, KeyContextT key_context) -> bool {
  return this->EraseImpl(lookup_key, key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto MapBase<InputKeyT, InputValueT, InputKeyContextT>::Clear() -> void {
  this->ClearImpl();
}

template <typename InputKeyT, typename InputValueT, ssize_t SmallSize,
          typename InputKeyContextT>
auto Map<InputKeyT, InputValueT, SmallSize, InputKeyContextT>::Reset() -> void {
  this->ResetImpl();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_MAP_H_
