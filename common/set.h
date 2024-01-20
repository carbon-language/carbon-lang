// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SET_H_
#define CARBON_COMMON_SET_H_

#include "common/check.h"
#include "common/raw_hashtable.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// Forward declarations to resolve cyclic references.
template <typename KeyT>
class SetView;
template <typename KeyT>
class SetBase;
template <typename KeyT, ssize_t MinSmallSize>
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
template <typename InputKeyT>
class SetView : RawHashtable::ViewImpl<InputKeyT> {
  using ImplT = RawHashtable::ViewImpl<InputKeyT>;

 public:
  using KeyT = typename ImplT::KeyT;

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
  SetView(SetView<std::remove_const_t<KeyT>> other_view)
    requires(!std::same_as<KeyT, std::remove_const_t<KeyT>>)
      : ImplT(other_view) {}

  // Tests whether a key is present in the set.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  // Lookup a key in the set.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult;

  // Run the provided callback for every key in the set.
  template <typename CallbackT>
  void ForEach(CallbackT callback);

  // Count the probed keys. This routine is purely informational and for use in
  // benchmarking or logging of performance anomalies. Its returns have no
  // semantic guarantee at all.
  auto CountProbedKeys() -> ssize_t { return ImplT::CountProbedKeys(); }

 private:
  template <typename SetKeyT, ssize_t MinSmallSize>
  friend class Set;
  friend class SetBase<KeyT>;
  friend class SetView<const KeyT>;

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
template <typename InputKeyT>
class SetBase : protected RawHashtable::BaseImpl<InputKeyT> {
  using ImplT = RawHashtable::BaseImpl<InputKeyT>;

 public:
  using KeyT = typename ImplT::KeyT;
  using ViewT = SetView<KeyT>;
  using LookupResult = typename ViewT::LookupResult;

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
  operator SetView<const KeyT>() const { return ViewT(*this); }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

  // Convenience forwarder to the view type.
  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupResult {
    return ViewT(*this).Lookup(lookup_key);
  }

  // Convenience forwarder to the view type.
  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
  }

  // Convenience forwarder to the view type.
  auto CountProbedKeys() const -> ssize_t {
    return ViewT(*this).CountProbedKeys();
  }

  // Insert a key into the set. If the key is already present, no insertion is
  // performed and that present key is available in the result. Otherwise a new
  // key is inserted and constructed from the argument and available in the
  // result.
  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key) -> InsertResult;

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
  auto Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb) -> InsertResult
    requires std::invocable<InsertCallbackT, LookupKeyT, void*>;

  // Erase a key from the set.
  template <typename LookupKeyT>
  auto Erase(LookupKeyT lookup_key) -> bool;

  // Clear all key/value pairs from the set but leave the underlying hashtable
  // allocated and in place.
  void Clear();

 protected:
  using ImplT::ImplT;
};

// A data structure for a set of keys.
//
// This set also supports small size optimization (or "SSO"). The provided
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
// with largely small strings, but if some or many strings are large heap
// allocations the performance of hashtable routines may be unacceptably bad and
// another data structure or key design is likely preferable.
//
// Note that this type should typically not appear on API boundaries; either
// `SetBase` or `SetView` should be used instead.
template <typename InputKeyT, ssize_t SmallSize = 0>
class Set : public RawHashtable::TableImpl<SetBase<InputKeyT>, SmallSize> {
  using BaseT = SetBase<InputKeyT>;
  using ImplT = RawHashtable::TableImpl<BaseT, SmallSize>;

 public:
  using KeyT = InputKeyT;

  Set() = default;
  Set(const Set& arg) = default;
  template <ssize_t OtherMinSmallSize>
  explicit Set(const Set<KeyT, OtherMinSmallSize>& arg) : ImplT(arg) {}
  Set(Set&& arg) noexcept = default;
  template <ssize_t OtherMinSmallSize>
  explicit Set(Set<KeyT, OtherMinSmallSize>&& arg) : ImplT(std::move(arg)) {}

  // Reset the entire state of the hashtable to as it was when constructed,
  // throwing away any intervening allocations.
  void Reset();
};

template <typename InputKeyT>
template <typename LookupKeyT>
auto SetView<InputKeyT>::Contains(LookupKeyT lookup_key) const -> bool {
  return this->LookupEntry(lookup_key) != nullptr;
}

template <typename KT>
template <typename LookupKeyT>
auto SetView<KT>::Lookup(LookupKeyT lookup_key) const -> LookupResult {
  EntryT* entry = this->LookupEntry(lookup_key);
  if (!entry) {
    return LookupResult();
  }

  return LookupResult(entry->key());
}

template <typename KT>
template <typename CallbackT>
void SetView<KT>::ForEach(CallbackT callback) {
  this->ForEachEntry([callback](EntryT& entry) { callback(entry.key()); },
                     [](auto...) {});
}

template <typename KT>
template <typename LookupKeyT>
auto SetBase<KT>::Insert(LookupKeyT lookup_key) -> InsertResult {
  return Insert(lookup_key, [](LookupKeyT lookup_key, void* key_storage) {
    new (key_storage) KeyT(std::move(lookup_key));
  });
}

template <typename KT>
template <typename LookupKeyT, typename InsertCallbackT>
auto SetBase<KT>::Insert(LookupKeyT lookup_key, InsertCallbackT insert_cb)
    -> InsertResult
  requires std::invocable<InsertCallbackT, LookupKeyT, void*>
{
  auto [entry, inserted] = this->InsertImpl(lookup_key);
  CARBON_DCHECK(entry) << "Should always result in a valid index.";

  if (LLVM_LIKELY(!inserted)) {
    return InsertResult(false, entry->key());
  }

  insert_cb(lookup_key, static_cast<void*>(&entry->key_storage));
  return InsertResult(true, entry->key());
}

template <typename KeyT>
template <typename LookupKeyT>
auto SetBase<KeyT>::Erase(LookupKeyT lookup_key) -> bool {
  return this->EraseImpl(lookup_key);
}

template <typename KeyT>
void SetBase<KeyT>::Clear() {
  this->ClearImpl();
}

template <typename KeyT, ssize_t SmallSize>
void Set<KeyT, SmallSize>::Reset() {
  this->ResetImpl();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_SET_H_
