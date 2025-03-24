// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_H_
#define CARBON_COMMON_SET_H_

#include <concepts>
#include <type_traits>

#include "common/check.h"
#include "common/hashtable_key_context.h"
#include "common/raw_hashtable.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// Forward declarations to resolve cyclic references.
template <typename KeyT, typename KeyContextT>
class SetView;
template <typename KeyT, typename KeyContextT>
class SetBase;
template <typename KeyT, ssize_t SmallSize, typename KeyContextT>
class Set;

// A read-only view type for a set of keys.
//
// This view is a cheap-to-copy type that should be passed by value, but
// provides view or read-only reference semantics to the underlying set data
// structure.
//
// This should always be preferred to a `const`-ref parameter for the `SetBase`
// or `Set` type as it provides more flexibility and a cleaner API.
//
// Note that while this type is a read-only view, that applies to the underlying
// *set* data structure, not the individual entries stored within it. Those can
// be mutated freely as long as both the hashes and equality of the keys are
// preserved. If we applied a deep-`const` design here, it would prevent using
// this type in situations where the keys carry state (unhashed and not part of
// equality) that is mutated while the associative container is not. A view of
// immutable data can always be obtained by using `SetView<const T>`, and we
// enable conversions to more-const views. This mirrors the semantics of views
// like `std::span`.
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
template <typename InputKeyT, typename InputKeyContextT = DefaultKeyContext>
class SetView : RawHashtable::ViewImpl<InputKeyT, void, InputKeyContextT> {
  using ImplT = RawHashtable::ViewImpl<InputKeyT, void, InputKeyContextT>;

 public:
  using KeyT = typename ImplT::KeyT;
  using KeyContextT = typename ImplT::KeyContextT;
  using MetricsT = typename ImplT::MetricsT;

  // This type represents the result of lookup operations. It encodes whether
  // the lookup was a success as well as accessors for the key.
  class LookupResult {
   public:
    LookupResult() = default;
    explicit LookupResult(KeyT& key) : key_(&key) {}

    explicit operator bool() const { return key_ != nullptr; }

    auto key() const -> KeyT& { return *key_; }

   private:
    KeyT* key_ = nullptr;
  };

  // Enable implicit conversions that add `const`-ness to the key type.
  // NOLINTNEXTLINE(google-explicit-constructor)
  SetView(const SetView<std::remove_const_t<KeyT>, KeyContextT>& other_view)
    requires(!std::same_as<KeyT, std::remove_const_t<KeyT>>)
      : ImplT(other_view) {}

  // Tests whether a key is present in the set.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key,
                KeyContextT key_context = KeyContextT()) const -> bool;

  // Lookup a key in the set.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key,
              KeyContextT key_context = KeyContextT()) const -> LookupResult;

  // Run the provided callback for every key in the set.
  template <typename CallbackT>
  auto ForEach(CallbackT callback) -> void
    requires(std::invocable<CallbackT, KeyT&>);

  // This routine is relatively inefficient and only intended for use in
  // benchmarking or logging of performance anomalies. The specific metrics
  // returned have no specific guarantees beyond being informative in
  // benchmarks.
  auto ComputeMetrics(KeyContextT key_context = KeyContextT()) -> MetricsT {
    return ImplT::ComputeMetricsImpl(key_context);
  }

 private:
  template <typename SetKeyT, ssize_t SmallSize, typename KeyContextT>
  friend class Set;
  friend class SetBase<KeyT, KeyContextT>;
  friend class SetView<const KeyT, KeyContextT>;

  using EntryT = typename ImplT::EntryT;

  SetView() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): Implicit by design.
  SetView(ImplT base) : ImplT(base) {}
  SetView(ssize_t size, RawHashtable::Storage* storage)
      : ImplT(size, storage) {}
};

// A base class for a `Set` type that remains mutable while type-erasing the
// `SmallSize` (SSO) template parameter.
//
// A pointer or reference to this type is the preferred way to pass a mutable
// handle to a `Set` type across API boundaries as it avoids encoding specific
// SSO sizing information while providing a near-complete mutable API.
template <typename InputKeyT, typename InputKeyContextT>
class SetBase
    : protected RawHashtable::BaseImpl<InputKeyT, void, InputKeyContextT> {
 protected:
  using ImplT = RawHashtable::BaseImpl<InputKeyT, void, InputKeyContextT>;

 public:
  using KeyT = typename ImplT::KeyT;
  using KeyContextT = typename ImplT::KeyContextT;
  using ViewT = SetView<KeyT, KeyContextT>;
  using LookupResult = typename ViewT::LookupResult;
  using MetricsT = typename ImplT::MetricsT;

  // The result type for insertion operations both indicates whether an insert
  // was needed (as opposed to the key already being in the set), and provides
  // access to the key.
  class InsertResult {
   public:
    InsertResult() = default;
    explicit InsertResult(bool inserted, KeyT& key)
        : key_(&key), inserted_(inserted) {}

    auto is_inserted() const -> bool { return inserted_; }

    auto key() const -> KeyT& { return *key_; }

   private:
    KeyT* key_;
    bool inserted_;
  };

  // Implicitly convertible to the relevant view type.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return this->view_impl(); }

  // We can't chain the above conversion with the conversions on `ViewT` to add
  // const, so explicitly support adding const to produce a view here.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator SetView<const KeyT, KeyContextT>() const { return ViewT(*this); }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key,
                KeyContextT key_context = KeyContextT()) const -> bool {
    return ViewT(*this).Contains(lookup_key, key_context);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key,
              KeyContextT key_context = KeyContextT()) const -> LookupResult {
    return ViewT(*this).Lookup(lookup_key, key_context);
  }

  // Convenience forwarder to the view type.
  template <typename CallbackT>
  auto ForEach(CallbackT callback) -> void
    requires(std::invocable<CallbackT, KeyT&>)
  {
    return ViewT(*this).ForEach(callback);
  }

  // Convenience forwarder to the view type.
  auto ComputeMetrics(KeyContextT key_context = KeyContextT()) const
      -> MetricsT {
    return ViewT(*this).ComputeMetrics(key_context);
  }

  // Insert a key into the set. If the key is already present, no insertion is
  // performed and that present key is available in the result. Otherwise a new
  // key is inserted and constructed from the argument and available in the
  // result.
  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key, KeyContextT key_context = KeyContextT())
      -> InsertResult;

  // Insert a key into the map and call the provided callback if necessary to
  // produce a new key when no existing value is found.
  //
  // Example: `m.Insert(key_equivalent, [] { return real_key; });`
  //
  // The point of this function is when the lookup key is _different_from the
  // stored key. However, we don't restrict it in case that blocks generic
  // usage.
  template <typename LookupKeyT, typename KeyCallbackT>
  auto Insert(LookupKeyT lookup_key, KeyCallbackT key_cb,
              KeyContextT key_context = KeyContextT()) -> InsertResult
    requires(
        !std::same_as<KeyT, KeyCallbackT> &&
        std::convertible_to<decltype(std::declval<KeyCallbackT>()()), KeyT>);

  // Insert a key into the set and call the provided callback to allow in-place
  // construction of the key if not already present. The lookup key is passed
  // through to the callback so it needn't be captured and can be kept in a
  // register argument throughout.
  //
  // Example:
  // ```cpp
  //   m.Insert("widget", [](MyStringViewType lookup_key, void* key_storage) {
  //     new (key_storage) MyStringType(lookup_key);
  //   });
  // ```
  template <typename LookupKeyT, typename InsertCallbackT>
  auto Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb,
              KeyContextT key_context = KeyContextT()) -> InsertResult
    requires std::invocable<InsertCallbackT, LookupKeyT, void*>;

  // Grow the set to a specific allocation size.
  //
  // This will grow the set's hashtable if necessary for it to have an
  // allocation size of `target_alloc_size` which must be a power of two. Note
  // that this will not allow that many keys to be inserted, but a smaller
  // number based on the maximum load factor. If a specific number of insertions
  // need to be achieved without triggering growth, use the `GrowForInsertCount`
  // method.
  auto GrowToAllocSize(ssize_t target_alloc_size,
                       KeyContextT key_context = KeyContextT()) -> void;

  // Grow the set sufficiently to allow inserting the specified number of keys.
  auto GrowForInsertCount(ssize_t count,
                          KeyContextT key_context = KeyContextT()) -> void;

  // Erase a key from the set.
  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key, KeyContextT key_context = KeyContextT())
      -> bool;

  // Clear all key/value pairs from the set but leave the underlying hashtable
  // allocated and in place.
  auto Clear() -> void;

 protected:
  using ImplT::ImplT;
};

// A data structure for a set of keys.
//
// This set supports small size optimization (or "SSO"). The provided
// `SmallSize` type parameter indicates the size of an embedded buffer for
// storing sets small enough to fit. The default is zero, which always allocates
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
// `SetBase` or `SetView` should be used instead.
template <typename InputKeyT, ssize_t SmallSize = 0,
          typename InputKeyContextT = DefaultKeyContext>
class Set : public RawHashtable::TableImpl<SetBase<InputKeyT, InputKeyContextT>,
                                           SmallSize> {
  using BaseT = SetBase<InputKeyT, InputKeyContextT>;
  using ImplT = RawHashtable::TableImpl<BaseT, SmallSize>;

 public:
  using KeyT = typename BaseT::KeyT;

  Set() = default;
  Set(const Set& arg) = default;
  Set(Set&& arg) noexcept = default;
  auto operator=(const Set& arg) -> Set& = default;
  auto operator=(Set&& arg) noexcept -> Set& = default;

  // Reset the entire state of the hashtable to as it was when constructed,
  // throwing away any intervening allocations.
  auto Reset() -> void;
};

template <typename InputKeyT, typename InputKeyContextT>
template <typename LookupKeyT>
auto SetView<InputKeyT, InputKeyContextT>::Contains(
    LookupKeyT lookup_key, KeyContextT key_context) const -> bool {
  return this->LookupEntry(lookup_key, key_context) != nullptr;
}

template <typename InputKeyT, typename InputKeyContextT>
template <typename LookupKeyT>
auto SetView<InputKeyT, InputKeyContextT>::Lookup(LookupKeyT lookup_key,
                                                  KeyContextT key_context) const
    -> LookupResult {
  EntryT* entry = this->LookupEntry(lookup_key, key_context);
  if (!entry) {
    return LookupResult();
  }

  return LookupResult(entry->key());
}

template <typename InputKeyT, typename InputKeyContextT>
template <typename CallbackT>
auto SetView<InputKeyT, InputKeyContextT>::ForEach(CallbackT callback) -> void
  requires(std::invocable<CallbackT, KeyT&>)
{
  this->ForEachEntry([callback](EntryT& entry) { callback(entry.key()); },
                     [](auto...) {});
}

template <typename InputKeyT, typename InputKeyContextT>
template <typename LookupKeyT>
auto SetBase<InputKeyT, InputKeyContextT>::Insert(LookupKeyT lookup_key,
                                                  KeyContextT key_context)
    -> InsertResult {
  return Insert(
      lookup_key,
      [](LookupKeyT lookup_key, void* key_storage) {
        new (key_storage) KeyT(std::move(lookup_key));
      },
      key_context);
}

template <typename InputKeyT, typename InputKeyContextT>
template <typename LookupKeyT, typename KeyCallbackT>
auto SetBase<InputKeyT, InputKeyContextT>::Insert(LookupKeyT lookup_key,
                                                  KeyCallbackT key_cb,
                                                  KeyContextT key_context)
    -> InsertResult
  requires(!std::same_as<KeyT, KeyCallbackT> &&
           std::convertible_to<decltype(std::declval<KeyCallbackT>()()), KeyT>)
{
  return Insert(
      lookup_key,
      [&key_cb](LookupKeyT /*lookup_key*/, void* key_storage) {
        new (key_storage) KeyT(key_cb());
      },
      key_context);
}

template <typename InputKeyT, typename InputKeyContextT>
template <typename LookupKeyT, typename InsertCallbackT>
auto SetBase<InputKeyT, InputKeyContextT>::Insert(LookupKeyT lookup_key,
                                                  InsertCallbackT insert_cb,
                                                  KeyContextT key_context)
    -> InsertResult
  requires std::invocable<InsertCallbackT, LookupKeyT, void*>
{
  auto [entry, inserted] = this->InsertImpl(lookup_key, key_context);
  CARBON_DCHECK(entry, "Should always result in a valid index.");

  if (LLVM_LIKELY(!inserted)) {
    return InsertResult(false, entry->key());
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage));
  return InsertResult(true, entry->key());
}

template <typename InputKeyT, typename InputKeyContextT>
auto SetBase<InputKeyT, InputKeyContextT>::GrowToAllocSize(
    ssize_t target_alloc_size, KeyContextT key_context) -> void {
  this->GrowToAllocSizeImpl(target_alloc_size, key_context);
}

template <typename InputKeyT, typename InputKeyContextT>
auto SetBase<InputKeyT, InputKeyContextT>::GrowForInsertCount(
    ssize_t count, KeyContextT key_context) -> void {
  this->GrowForInsertCountImpl(count, key_context);
}

template <typename InputKeyT, typename InputKeyContextT>
template <typename LookupKeyT>
auto SetBase<InputKeyT, InputKeyContextT>::Erase(LookupKeyT lookup_key,
                                                 KeyContextT key_context)
    -> bool {
  return this->EraseImpl(lookup_key, key_context);
}

template <typename InputKeyT, typename InputKeyContextT>
auto SetBase<InputKeyT, InputKeyContextT>::Clear() -> void {
  this->ClearImpl();
}

template <typename InputKeyT, ssize_t SmallSize, typename InputKeyContextT>
auto Set<InputKeyT, SmallSize, InputKeyContextT>::Reset() -> void {
  this->ResetImpl();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
