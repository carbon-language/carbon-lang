// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_H_
#define CARBON_COMMON_RAW_HASHTABLE_H_

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/hashing.h"
#include "common/raw_hashtable_metadata_group.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"

// A namespace collecting a set of low-level utilities for building hashtable
// data structures. These should only be used as implementation details of
// higher-level data-structure APIs.
//
// The utilities here use the `hashtable_key_context.h` provided `KeyContext` to
// support the necessary hashtable operations on keys: hashing and comparison.
// This also serves as the customization point for hashtables built on this
// infrastructure for those operations. See that header file for details.
//
// These utilities support hashtables following a *specific* API design pattern,
// and using Small-Size Optimization, or "SSO", when desired. We expect there to
// be three layers to any hashtable design:
//
// - A *view* type: a read-only view of the hashtable contents. This type should
//   be a value type and is expected to be passed by-value in APIs. However, it
//   will have `const`-reference semantics, much like a `std::string_view`. Note
//   that the *entries* will continue to be mutable, it is only the *table* that
//   is read-only.
//
// - A *base* type: a base class type of the actual hashtable, which allows
//   almost all mutable operations but erases any specific SSO buffer size.
//   Because this is a base of the actual hash table, it is designed to be
//   passed as a non-`const` reference or pointer.
//
// - A *table* type: the actual hashtable which derives from the base type and
//   adds any desired SSO storage buffer. Beyond the physical storage, it also
//   allows resetting the table to its initial state & allocated size, as well
//   as copying and moving the table.
//
// For complete examples of the API design, see `set.h` for a hashtable-based
// set data structure, and `map.h` for a hashtable-based map data structure.
//
// The hashtable design implemented here has several key invariants and design
// elements that are essential to all three of the types above and the
// functionality they provide.
//
// - The underlying hashtable uses [open addressing], a power-of-two table size,
//   and quadratic probing rather than closed addressing and chaining.
//
//   [open addressing]: https://en.wikipedia.org/wiki/Open_addressing
//
// - Each _slot_ in the table corresponds to a key, a value, and one byte of
//   metadata. Each _entry_ is a key and value. The key and value for an entry
//   are stored together.
//
// - The allocated storage is organized into an array of metadata bytes followed
//   by an array of entry storage.
//
// - The metadata byte corresponding to each entry marks that entry is either
//   empty, deleted, or present. When present, a 7-bit tag is also stored using
//   another 7 bits from the hash of the entry key.
//
// - The storage for an entry is an internal type that should not be exposed to
//   users, and instead only the underlying keys and values.
//
// - The hash addressing and probing occurs over *groups* of slots rather than
//   individual entries. When inserting a new entry, it can be added to the
//   group it hashes to as long it is not full, and can even replace a slot with
//   a tombstone indicating a previously deleted entry. Only when the group is
//   full will it look at the next group in the probe sequence. As a result,
//   there may be entries in a group where a different group is the start of
//   that entry's probe sequence. Also, when performing a lookup, every group in
//   the probe sequence must be inspected for the lookup key until it is found
//   or the group has an empty slot.
//
// - Groups are scanned rapidly using the one-byte metadata for each entry in
//   the group and CPU instructions that allow comparing all of the metadata for
//   a group in parallel. For more details on the metadata group encoding and
//   scanning, see `raw_hashtable_metadata_group.h`.
//
// - `GroupSize` is a platform-specific relatively small power of two that fits
//   in some hardware register. However, `MaxGroupSize` is provided as a
//   portable max that is also a power of two. The table storage, whether
//   provided by an SSO buffer or allocated, is required to be a multiple of
//   `MaxGroupSize` to keep the requirement portable but sufficient for all
//   platforms.
//
// - There is *always* an allocated table of some multiple of `MaxGroupSize`.
//   This allows accesses to be branchless. When heap allocated, we pro-actively
//   allocate at least a minimum heap size table. When there is a small-size
//   optimization (SSO) buffer, that provides the initial allocation.
//
// - The table performs a minimal amount of bookkeeping that limits the APIs it
//   can support:
//    - `alloc_size` is the size of the table *allocated* (not *used*), and is
//       always a power of 2 at least as big as `MinAllocatedSize`.
//    - `storage` is a pointer to the storage for the `alloc_size` slots of the
//       table, and never null.
//    - `small_alloc_size` is the maximum `alloc_size` where the table is stored
//       in the object itself instead of separately on the heap. In this case,
//       `storage` points to `small_storage_`.
//    - `growth_budget` is the number of entries that may be added before the
//       table allocation is doubled. It is always
//       `GrowthThresholdForAllocSize(alloc_size)` minus the number of
//       non-empty (filled or deleted) slots. If it ever falls to 0, the table
//       is grown to keep it greater than 0.
//   There is also the "moved-from" state where the table may only be
//   reinitialized or destroyed where the `alloc_size` is 0 and `storage` is
//   null. Since it doesn't track the exact number of filled entries in a table,
//   it doesn't support a container-style `size` API.
//
// - There is no direct iterator support because of the complexity of embedding
//   the group-based metadata scanning into an iterator model. Instead, there is
//   just a for-each method that is passed a lambda to observe all entries. The
//   order of this observation is also not guaranteed.
namespace Carbon::RawHashtable {

// Which prefetch strategies to enable can be controlled via macros to enable
// doing experiments.
//
// Currently, benchmarking on both modern AMD and ARM CPUs seems to indicate
// that the entry group prefetching is more beneficial than metadata, but that
// benefit is degraded when enabling them both. This determined our current
// default of no metadata prefetch but enabled entry group prefetch.
//
// Override these by defining them as part of the build explicitly to either `0`
// or `1`. If left undefined, the defaults will be supplied.
#ifndef CARBON_ENABLE_PREFETCH_METADATA
#define CARBON_ENABLE_PREFETCH_METADATA 0
#endif
#ifndef CARBON_ENABLE_PREFETCH_ENTRY_GROUP
#define CARBON_ENABLE_PREFETCH_ENTRY_GROUP 1
#endif

// If allocating storage, allocate a minimum of one cacheline of group metadata
// or a minimum of one group, whichever is larger.
constexpr ssize_t MinAllocatedSize = std::max<ssize_t>(64, MaxGroupSize);

// An entry in the hashtable storage of a `KeyT` and `ValueT` object.
//
// Allows manual construction, destruction, and access to these values so we can
// create arrays af the entries prior to populating them with actual keys and
// values.
template <typename KeyT, typename ValueT>
struct StorageEntry {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT> &&
      std::is_trivially_destructible_v<ValueT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT> &&
      std::is_trivially_move_constructible_v<ValueT>;

  auto key() const -> const KeyT& {
    // Ensure we don't need more alignment than available. Inside a method body
    // to apply to the complete type.
    static_assert(
        alignof(StorageEntry) <= MinAllocatedSize,
        "The minimum allocated size turns into the alignment of our array of "
        "storage entries as they follow the metadata byte array.");

    return *std::launder(reinterpret_cast<const KeyT*>(&key_storage));
  }
  auto key() -> KeyT& {
    return const_cast<KeyT&>(const_cast<const StorageEntry*>(this)->key());
  }

  auto value() const -> const ValueT& {
    return *std::launder(reinterpret_cast<const ValueT*>(&value_storage));
  }
  auto value() -> ValueT& {
    return const_cast<ValueT&>(const_cast<const StorageEntry*>(this)->value());
  }

  // We handle destruction and move manually as we only want to expose distinct
  // `KeyT` and `ValueT` subobjects to user code that may need to do in-place
  // construction. As a consequence, this struct only provides the storage and
  // we have to manually manage the construction, move, and destruction of the
  // objects.
  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key().~KeyT();
    value().~ValueT();
  }

  auto CopyFrom(const StorageEntry& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(entry.key());
      new (&value_storage) ValueT(entry.value());
    }
  }

  // Move from an expiring entry and destroy that entry's key and value.
  // Optimizes to directly use `memcpy` when correct.
  auto MoveFrom(StorageEntry&& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(std::move(entry.key()));
      entry.key().~KeyT();
      new (&value_storage) ValueT(std::move(entry.value()));
      entry.value().~ValueT();
    }
  }

  alignas(KeyT) std::byte key_storage[sizeof(KeyT)];
  alignas(ValueT) std::byte value_storage[sizeof(ValueT)];
};

// A specialization of the storage entry for sets without a distinct value type.
// Somewhat duplicative with the key-value version, but C++ specialization makes
// doing better difficult.
template <typename KeyT>
struct StorageEntry<KeyT, void> {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT>;

  auto key() const -> const KeyT& {
    // Ensure we don't need more alignment than available.
    static_assert(
        alignof(StorageEntry) <= MinAllocatedSize,
        "The minimum allocated size turns into the alignment of our array of "
        "storage entries as they follow the metadata byte array.");

    return *std::launder(reinterpret_cast<const KeyT*>(&key_storage));
  }
  auto key() -> KeyT& {
    return const_cast<KeyT&>(const_cast<const StorageEntry*>(this)->key());
  }

  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key().~KeyT();
  }

  auto CopyFrom(const StorageEntry& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(entry.key());
    }
  }

  auto MoveFrom(StorageEntry&& entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(this, &entry, sizeof(StorageEntry));
    } else {
      new (&key_storage) KeyT(std::move(entry.key()));
      entry.key().~KeyT();
    }
  }

  alignas(KeyT) std::byte key_storage[sizeof(KeyT)];
};

struct Metrics {
  // How many keys are present in the table.
  ssize_t key_count = 0;
  // How many slots of the table are reserved due to deleted markers required to
  // preserve probe sequences.
  ssize_t deleted_count = 0;
  // How many bytes of allocated storage are used by the table. Note, does not
  // include the table object or any small-size buffer.
  ssize_t storage_bytes = 0;

  // How many keys have required probing beyond the initial group. These are the
  // keys with a probe distance > 0.
  ssize_t probed_key_count = 0;
  // The probe distance averaged over every key. If every key is in its initial
  // group, this will be zero as no keys will have a larger probe distance. In
  // general, we want this to be as close to zero as possible.
  double probe_avg_distance = 0.0;
  // The maximum probe distance found for a single key in the table.
  ssize_t probe_max_distance = 0;
  // The average number of probing comparisons required to locate a specific key
  // in the table. This is how many comparisons are required *before* the key is
  // located, or the *failed* comparisons. We always have to do one successful
  // comparison at the end. This successful comparison isn't counted because
  // that focuses this metric on the overhead the table is introducing, and
  // keeps a "perfect" table with an average of `0.0` here similar to the
  // perfect average of `0.0` average probe distance.
  double probe_avg_compares = 0.0;
  // The maximum number of probing comparisons required to locate a specific
  // key in the table.
  ssize_t probe_max_compares = 0;
};

// A placeholder empty type used to model pointers to the allocated buffer of
// storage.
//
// The allocated storage doesn't have a meaningful static layout -- it consists
// of an array of metadata groups followed by an array of storage entries.
// However, we want to be able to mark pointers to this and so use pointers to
// this placeholder type as that signifier.
//
// This is a complete, empty type so that it can be used as a base class of a
// specific concrete storage type for compile-time sized storage.
struct Storage {};

// Forward declaration to support friending, see the definition below.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
class BaseImpl;

// Implementation helper for defining a read-only view type for a hashtable.
//
// A specific user-facing hashtable view type should derive privately from this
// type, and forward the implementation of its interface to functions in this
// type.
//
// The methods available to user-facing hashtable types are `protected`, and
// where they are expected to directly map to a public API, named with an
// `Impl`. The suffix naming ensures types don't `using` in these low-level APIs
// but declare their own and implement them by forwarding to these APIs. We
// don't want users to have to read these implementation details to understand
// their container's API, so none of these methods should be `using`-ed into the
// user facing types.
//
// Some of the types are just convenience aliases and aren't important to
// surface as part of the user-facing type API for readers and so those are
// reasonable to add via a `using`.
//
// Some methods are used by other parts of the raw hashtable implementation.
// Those are kept `private` and where necessary the other components of the raw
// hashtable implementation are friended to give access to them.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
class ViewImpl {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using KeyContextT = InputKeyContextT;
  using EntryT = StorageEntry<KeyT, ValueT>;
  using MetricsT = Metrics;

  friend class BaseImpl<KeyT, ValueT, KeyContextT>;
  template <typename InputBaseT, ssize_t SmallSize>
  friend class TableImpl;

  // Make more-`const` types friends to enable conversions that add `const`.
  friend class ViewImpl<const KeyT, ValueT, KeyContextT>;
  friend class ViewImpl<KeyT, const ValueT, KeyContextT>;
  friend class ViewImpl<const KeyT, const ValueT, KeyContextT>;

  ViewImpl() = default;

  // Support adding `const` to either key or value type of some other view.
  template <typename OtherKeyT, typename OtherValueT>
  // NOLINTNEXTLINE(google-explicit-constructor)
  ViewImpl(ViewImpl<OtherKeyT, OtherValueT, KeyContextT> other_view)
    requires(std::same_as<KeyT, OtherKeyT> ||
             std::same_as<KeyT, const OtherKeyT>) &&
                (std::same_as<ValueT, OtherValueT> ||
                 std::same_as<ValueT, const OtherValueT>)
      : alloc_size_(other_view.alloc_size_), storage_(other_view.storage_) {}

  // Looks up an entry in the hashtable and returns its address or null if not
  // present.
  template <typename LookupKeyT>
  auto LookupEntry(LookupKeyT lookup_key, KeyContextT key_context) const
      -> EntryT*;

  // Calls `entry_callback` for each entry in the hashtable. All the entries
  // within a specific group are visited first, and then `group_callback` is
  // called on the group itself. The `group_callback` is typically only used by
  // the internals of the hashtable.
  template <typename EntryCallbackT, typename GroupCallbackT>
  auto ForEachEntry(EntryCallbackT entry_callback,
                    GroupCallbackT group_callback) const -> void;

  // Returns a collection of informative metrics on the the current state of the
  // table, useful for performance analysis. These include relatively slow to
  // compute metrics requiring deep inspection of the table's state.
  auto ComputeMetricsImpl(KeyContextT key_context) const -> MetricsT;

 private:
  ViewImpl(ssize_t alloc_size, Storage* storage)
      : alloc_size_(alloc_size), storage_(storage) {}

  // Computes the offset from the metadata array to the entries array for a
  // given size. This is trivial, but we use this routine to enforce invariants
  // on the sizes.
  static constexpr auto EntriesOffset(ssize_t alloc_size) -> ssize_t {
    CARBON_DCHECK(llvm::isPowerOf2_64(alloc_size),
                  "Size must be a power of two for a hashed buffer!");
    // The size is always a power of two. We prevent any too-small sizes so it
    // being a power of two provides the needed alignment. As a result, the
    // offset is exactly the size. We validate this here to catch alignment bugs
    // early.
    CARBON_DCHECK(static_cast<uint64_t>(alloc_size) ==
                  llvm::alignTo<alignof(EntryT)>(alloc_size));
    return alloc_size;
  }

  // Compute the allocated table's byte size.
  static constexpr auto AllocByteSize(ssize_t alloc_size) -> ssize_t {
    return EntriesOffset(alloc_size) + sizeof(EntryT) * alloc_size;
  }

  auto metadata() const -> uint8_t* {
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto entries() const -> EntryT* {
    return reinterpret_cast<EntryT*>(reinterpret_cast<std::byte*>(storage_) +
                                     EntriesOffset(alloc_size_));
  }

  // Prefetch the metadata prior to probing. This is to overlap any of the
  // memory access latency we can with the hashing of a key or other
  // latency-bound operation prior to probing.
  auto PrefetchMetadata() const -> void {
    if constexpr (CARBON_ENABLE_PREFETCH_METADATA) {
      // Prefetch with a "low" temporal locality as we're primarily expecting a
      // brief use of the metadata and then to return to application code.
      __builtin_prefetch(metadata(), /*read*/ 0, /*low-locality*/ 1);
    }
  }

  // Prefetch an entry. This prefetches for read as it is primarily expected to
  // be used in the probing path, and writing afterwards isn't especially slowed
  // down. We don't want to synthesize writes unless we *know* we're going to
  // write.
  static auto PrefetchEntryGroup(const EntryT* entry_group) -> void {
    if constexpr (CARBON_ENABLE_PREFETCH_ENTRY_GROUP) {
      // Prefetch with a "low" temporal locality as we're primarily expecting a
      // brief use of the entries and then to return to application code.
      __builtin_prefetch(entry_group, /*read*/ 0, /*low-locality*/ 1);
    }
  }

  ssize_t alloc_size_;
  Storage* storage_;
};

// Implementation helper for defining a read-write base type for a hashtable
// that type-erases any SSO buffer.
//
// A specific user-facing hashtable base type should derive using *`protected`*
// inheritance from this type, and forward the implementation of its interface
// to functions in this type.
//
// Other than the use of `protected` inheritance, the patterns for this type,
// and how to build user-facing hashtable base types from it, mirror those of
// `ViewImpl`. See its documentation for more details.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
class BaseImpl {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using KeyContextT = InputKeyContextT;
  using ViewImplT = ViewImpl<KeyT, ValueT, KeyContextT>;
  using EntryT = typename ViewImplT::EntryT;
  using MetricsT = typename ViewImplT::MetricsT;

  BaseImpl(int small_alloc_size, Storage* small_storage)
      : small_alloc_size_(small_alloc_size) {
    CARBON_CHECK(small_alloc_size >= 0);
    Construct(small_storage);
  }
  // Only used for copying and moving, and leaves storage uninitialized.
  BaseImpl(ssize_t alloc_size, int growth_budget, int small_alloc_size)
      : view_impl_(alloc_size, nullptr),
        growth_budget_(growth_budget),
        small_alloc_size_(small_alloc_size) {}

  // Destruction must be handled by the table where it can destroy entries in
  // any small buffer, so make the base destructor protected but defaulted here.
  ~BaseImpl() = default;

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewImplT() const { return view_impl(); }

  auto view_impl() const -> ViewImplT { return view_impl_; }

  // Looks up the provided key in the hashtable. If found, returns a pointer to
  // that entry and `false`.
  //
  // If not found, will locate an empty entry for inserting into, set the
  // metadata for that entry, and return a pointer to the entry and `true`. When
  // necessary, this will grow the hashtable to cause there to be sufficient
  // empty entries.
  template <typename LookupKeyT>
  auto InsertImpl(LookupKeyT lookup_key, KeyContextT key_context)
      -> std::pair<EntryT*, bool>;

  // Grow the table to specific allocation size.
  //
  // This will grow the the table if necessary for it to have an allocation size
  // of `target_alloc_size` which must be a power of two. Note that this will
  // not allow that many keys to be inserted into the hashtable, but a smaller
  // number based on the load factor. If a specific number of insertions need to
  // be achieved without triggering growth, use the `GrowForInsertCountImpl`
  // method.
  auto GrowToAllocSizeImpl(ssize_t target_alloc_size, KeyContextT key_context)
      -> void;

  // Grow the table to allow inserting the specified number of keys.
  auto GrowForInsertCountImpl(ssize_t count, KeyContextT key_context) -> void;

  // Looks up the entry in the hashtable, and if found destroys the entry and
  // returns `true`. If not found, returns `false`.
  //
  // Does not release any memory, just leaves a tombstone behind so this entry
  // cannot be found and the slot can in theory be reused.
  template <typename LookupKeyT>
  auto EraseImpl(LookupKeyT lookup_key, KeyContextT key_context) -> bool;

  // Erases all entries in the hashtable but leaves the allocated storage.
  auto ClearImpl() -> void;

 private:
  template <typename InputBaseT, ssize_t SmallSize>
  friend class TableImpl;

  static constexpr ssize_t Alignment = std::max<ssize_t>(
      alignof(MetadataGroup), alignof(StorageEntry<KeyT, ValueT>));

  // Implementation of inline small storage for the provided key type, value
  // type, and small size. Specialized for a zero small size to be an empty
  // struct.
  template <ssize_t SmallSize>
  struct SmallStorage : Storage {
    alignas(Alignment) uint8_t metadata[SmallSize];
    mutable StorageEntry<KeyT, ValueT> entries[SmallSize];
  };
  // Specialized storage with no inline buffer to avoid any extra alignment.
  template <>
  struct SmallStorage<0> {};

  static auto Allocate(ssize_t alloc_size) -> Storage*;
  static auto Deallocate(Storage* storage, ssize_t alloc_size) -> void;

  auto growth_budget() const -> ssize_t { return growth_budget_; }
  auto alloc_size() const -> ssize_t { return view_impl_.alloc_size_; }
  auto alloc_size() -> ssize_t& { return view_impl_.alloc_size_; }
  auto storage() const -> Storage* { return view_impl_.storage_; }
  auto storage() -> Storage*& { return view_impl_.storage_; }
  auto metadata() const -> uint8_t* { return view_impl_.metadata(); }
  auto entries() const -> EntryT* { return view_impl_.entries(); }
  auto small_alloc_size() const -> ssize_t {
    return static_cast<unsigned>(small_alloc_size_);
  }
  auto is_small() const -> bool {
    CARBON_DCHECK(alloc_size() >= small_alloc_size());
    return alloc_size() == small_alloc_size();
  }

  // Wrapper to call `ViewImplT::PrefetchStorage`, see that method for details.
  auto PrefetchStorage() const -> void { view_impl_.PrefetchMetadata(); }

  auto Construct(Storage* small_storage) -> void;
  auto Destroy() -> void;
  auto CopySlotsFrom(const BaseImpl& arg) -> void;
  auto MoveFrom(BaseImpl&& arg, Storage* small_storage) -> void;

  auto InsertIntoEmpty(HashCode hash) -> EntryT*;

  static auto ComputeNextAllocSize(ssize_t old_alloc_size) -> ssize_t;
  static auto GrowthThresholdForAllocSize(ssize_t alloc_size) -> ssize_t;

  auto GrowToNextAllocSize(KeyContextT key_context) -> void;
  auto GrowAndInsert(HashCode hash, KeyContextT key_context) -> EntryT*;

  ViewImplT view_impl_;
  int growth_budget_;
  int small_alloc_size_;
};

// Implementation helper for defining a hashtable type with an SSO buffer.
//
// A specific user-facing hashtable should derive privately from this
// type, and forward the implementation of its interface to functions in this
// type. It should provide the corresponding user-facing hashtable base type as
// the `InputBaseT` type parameter (rather than a key/value pair), and this type
// will in turn derive from that provided base type. This allows derived-to-base
// conversion from the user-facing hashtable type to the user-facing hashtable
// base type. And it does so keeping the inheritance linear. The resulting
// linear inheritance hierarchy for a `Map<K, T>` type will look like:
//
//   Map<K, T>
//    ↓
//   TableImpl<MapBase<K, T>>
//    ↓
//   MapBase<K, T>
//    ↓
//   BaseImpl<K, T>
//
// Other than this inheritance technique, the patterns for this type, and how to
// build user-facing hashtable types from it, mirror those of `ViewImpl`. See
// its documentation for more details.
template <typename InputBaseT, ssize_t SmallSize>
class TableImpl : public InputBaseT {
 protected:
  using BaseT = InputBaseT;

  TableImpl() : BaseT(SmallSize, small_storage()) {}
  TableImpl(const TableImpl& arg);
  TableImpl(TableImpl&& arg) noexcept;
  auto operator=(const TableImpl& arg) -> TableImpl&;
  auto operator=(TableImpl&& arg) noexcept -> TableImpl&;
  ~TableImpl();

  // Resets the hashtable to its initial state, clearing all entries and
  // releasing all memory. If the hashtable had an SSO buffer, that is restored
  // as the storage. Otherwise, a minimum sized table storage is allocated.
  auto ResetImpl() -> void;

 private:
  using KeyT = BaseT::KeyT;
  using ValueT = BaseT::ValueT;
  using EntryT = BaseT::EntryT;
  using SmallStorage = BaseT::template SmallStorage<SmallSize>;

  auto small_storage() const -> Storage*;

  auto SetUpStorage() -> void;

  [[no_unique_address]] mutable SmallStorage small_storage_;
};

////////////////////////////////////////////////////////////////////////////////
//
// Only implementation details below this point.
//
////////////////////////////////////////////////////////////////////////////////

// Computes a seed that provides a small amount of entropy from ASLR where
// available with minimal cost. The priority is speed, and this computes the
// entropy in a way that doesn't require loading from memory, merely accessing
// entropy already available without accessing memory.
inline auto ComputeSeed() -> uint64_t {
  // A global variable whose address is used as a seed. This allows ASLR to
  // introduce some variation in hashtable ordering when enabled via the code
  // model for globals.
  extern volatile std::byte global_addr_seed;

  return reinterpret_cast<uint64_t>(&global_addr_seed);
}

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
  CARBON_DCHECK(llvm::isPowerOf2_64(size),
                "Size must be a power of two for a hashed buffer!");
  // Since `size` is a power of two, we can make sure the probes are less
  // than `size` by making the mask `size - 1`. We also mask off the low
  // bits so the probes are a multiple of the size of the groups of entries.
  return (size - 1) & ~GroupMask;
}

// This class handles building a sequence of probe indices from a given
// starting point, including both the quadratic growth and masking the index
// to stay within the bucket array size. The starting point doesn't need to be
// clamped to the size ahead of time (or even be positive), we will do it
// internally.
//
// For reference on quadratic probing:
// https://en.wikipedia.org/wiki/Quadratic_probing
//
// We compute the quadratic probe index incrementally, but we can also compute
// it mathematically and will check that the incremental result matches our
// mathematical expectation. We use the quadratic probing formula of:
//
//   p(start, step) = (start + (step + step^2) / 2) (mod size / GroupSize)
//
// However, we compute it incrementally and scale all the variables by the group
// size so it can be used as an index without an additional multiplication.
class ProbeSequence {
 public:
  ProbeSequence(ssize_t start, ssize_t size) {
    mask_ = ComputeProbeMaskFromSize(size);
    p_ = start & mask_;
#ifndef NDEBUG
    start_ = start & mask_;
    size_ = size;
#endif
  }

  auto Next() -> void {
    step_ += GroupSize;
    p_ = (p_ + step_) & mask_;
#ifndef NDEBUG
    // Verify against the quadratic formula we expect to be following by scaling
    // everything down by `GroupSize`.
    CARBON_DCHECK(
        (p_ / GroupSize) ==
            ((start_ / GroupSize +
              (step_ / GroupSize + (step_ / GroupSize) * (step_ / GroupSize)) /
                  2) %
             (size_ / GroupSize)),
        "Index in probe sequence does not match the expected formula.");
    CARBON_DCHECK(step_ < size_,
                  "We necessarily visit all groups, so we can't have more "
                  "probe steps than groups.");
#endif
  }

  auto index() const -> ssize_t { return p_; }

 private:
  ssize_t step_ = 0;
  size_t mask_;
  ssize_t p_;
#ifndef NDEBUG
  ssize_t start_;
  ssize_t size_;
#endif
};

// TODO: Evaluate keeping this outlined to see if macro benchmarks observe the
// same perf hit as micro benchmarks.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto ViewImpl<InputKeyT, InputValueT, InputKeyContextT>::LookupEntry(
    LookupKeyT lookup_key, KeyContextT key_context) const -> EntryT* {
  PrefetchMetadata();

  ssize_t local_size = alloc_size_;
  CARBON_DCHECK(local_size > 0);

  uint8_t* local_metadata = metadata();
  HashCode hash = key_context.HashKey(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();

  EntryT* local_entries = entries();

  // Walk through groups of entries using a quadratic probe starting from
  // `hash_index`.
  ProbeSequence s(hash_index, local_size);
  do {
    ssize_t group_index = s.index();

    // Load the group's metadata and prefetch the entries for this group. The
    // prefetch here helps hide key access latency while we're matching the
    // metadata.
    MetadataGroup g = MetadataGroup::Load(local_metadata, group_index);
    EntryT* group_entries = &local_entries[group_index];
    PrefetchEntryGroup(group_entries);

    // For each group, match the tag against the metadata to extract the
    // potentially matching entries within the group.
    auto metadata_matched_range = g.Match(tag);
    if (LLVM_LIKELY(metadata_matched_range)) {
      // If any entries in this group potentially match based on their metadata,
      // walk each candidate and compare its key to see if we have definitively
      // found a match.
      auto byte_it = metadata_matched_range.begin();
      auto byte_end = metadata_matched_range.end();
      do {
        EntryT* entry = byte_it.index_ptr(group_entries);
        if (LLVM_LIKELY(key_context.KeyEq(lookup_key, entry->key()))) {
          __builtin_assume(entry != nullptr);
          return entry;
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots as that indicates we're done probing -- no later probed index
    // could have a match.
    auto empty_byte_matched_range = g.MatchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      return nullptr;
    }

    s.Next();

    // We use a weird construct of an "unlikely" condition of `true`. The goal
    // is to get the compiler to not prioritize the back edge of the loop for
    // code layout, and in at least some tests this seems to be an effective
    // construct for achieving this.
  } while (LLVM_UNLIKELY(true));
}

// Note that we force inlining here because we expect to be called with lambdas
// that will in turn be inlined to form the loop body. We don't want function
// boundaries within the loop for performance, and recognizing the degree of
// simplification from inlining these callbacks may be difficult to
// automatically recognize.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename EntryCallbackT, typename GroupCallbackT>
[[clang::always_inline]] auto
ViewImpl<InputKeyT, InputValueT, InputKeyContextT>::ForEachEntry(
    EntryCallbackT entry_callback, GroupCallbackT group_callback) const
    -> void {
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();

  ssize_t local_size = alloc_size_;
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(local_metadata, group_index);
    auto present_matched_range = g.MatchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      entry_callback(local_entries[group_index + byte_index]);
    }

    group_callback(&local_metadata[group_index]);
  }
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto ViewImpl<InputKeyT, InputValueT, InputKeyContextT>::ComputeMetricsImpl(
    KeyContextT key_context) const -> Metrics {
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();
  ssize_t local_size = alloc_size_;

  Metrics metrics;

  // Compute the ones we can directly.
  metrics.deleted_count = llvm::count(
      llvm::ArrayRef(local_metadata, local_size), MetadataGroup::Deleted);
  metrics.storage_bytes = AllocByteSize(local_size);

  // We want to process present slots specially to collect metrics on their
  // probing behavior.
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(local_metadata, group_index);
    auto present_matched_range = g.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      ++metrics.key_count;
      ssize_t index = group_index + byte_index;
      HashCode hash =
          key_context.HashKey(local_entries[index].key(), ComputeSeed());
      auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
      ProbeSequence s(hash_index, local_size);
      metrics.probed_key_count +=
          static_cast<ssize_t>(s.index() != group_index);

      // For each probed key, go through the probe sequence to find both the
      // probe distance and how many comparisons are required.
      ssize_t distance = 0;
      ssize_t compares = 0;
      for (; s.index() != group_index; s.Next()) {
        auto probe_g = MetadataGroup::Load(local_metadata, s.index());
        auto probe_matched_range = probe_g.Match(tag);
        compares += std::distance(probe_matched_range.begin(),
                                  probe_matched_range.end());
        distance += 1;
      }

      auto probe_g = MetadataGroup::Load(local_metadata, s.index());
      auto probe_matched_range = probe_g.Match(tag);
      CARBON_CHECK(!probe_matched_range.empty());
      for (ssize_t match_index : probe_matched_range) {
        if (match_index >= byte_index) {
          // Note we only count the compares that will *fail* as part of
          // probing. The last successful compare isn't interesting, it is
          // always needed.
          break;
        }
        compares += 1;
      }
      metrics.probe_avg_distance += distance;
      metrics.probe_max_distance =
          std::max(metrics.probe_max_distance, distance);
      metrics.probe_avg_compares += compares;
      metrics.probe_max_compares =
          std::max(metrics.probe_max_compares, compares);
    }
  }
  if (metrics.key_count > 0) {
    metrics.probe_avg_compares /= metrics.key_count;
    metrics.probe_avg_distance /= metrics.key_count;
  }
  return metrics;
}

// TODO: Evaluate whether it is worth forcing this out-of-line given the
// reasonable ABI boundary it forms and large volume of code necessary to
// implement it.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::InsertImpl(
    LookupKeyT lookup_key, KeyContextT key_context)
    -> std::pair<EntryT*, bool> {
  CARBON_DCHECK(alloc_size() > 0);
  PrefetchStorage();

  uint8_t* local_metadata = metadata();

  HashCode hash = key_context.HashKey(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();

  // We re-purpose the empty control byte to signal no insert is needed to the
  // caller. This is guaranteed to not be a control byte we're inserting.
  // constexpr uint8_t NoInsertNeeded = Group::Empty;

  ssize_t group_with_deleted_index;
  MetadataGroup::MatchIndex deleted_match = {};

  EntryT* local_entries = entries();

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<EntryT*, bool> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    local_metadata[index] = tag | MetadataGroup::PresentMask;
    return {&local_entries[index], true};
  };

  for (ProbeSequence s(hash_index, alloc_size());; s.Next()) {
    ssize_t group_index = s.index();

    // Load the group's metadata and prefetch the entries for this group. The
    // prefetch here helps hide key access latency while we're matching the
    // metadata.
    auto g = MetadataGroup::Load(local_metadata, group_index);
    EntryT* group_entries = &local_entries[group_index];
    ViewImplT::PrefetchEntryGroup(group_entries);

    auto control_byte_matched_range = g.Match(tag);
    if (control_byte_matched_range) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        EntryT* entry = byte_it.index_ptr(group_entries);
        if (LLVM_LIKELY(key_context.KeyEq(lookup_key, entry->key()))) {
          return {entry, false};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (!deleted_match) {
      deleted_match = g.MatchDeleted();
      group_with_deleted_index = group_index;
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // no empty slots. In that case, we'll continue probing.
    auto empty_match = g.MatchEmpty();
    if (!empty_match) {
      continue;
    }
    // Ok, we've finished probing without finding anything and need to insert
    // instead.

    // If we found a deleted slot, we don't need the probe sequence to insert
    // so just bail. We want to ensure building up a table is fast so we
    // de-prioritize this a bit. In practice this doesn't have too much of an
    // effect.
    if (LLVM_UNLIKELY(deleted_match)) {
      return return_insert_at_index(group_with_deleted_index +
                                    deleted_match.index());
    }

    // We're going to need to grow by inserting into an empty slot. Check that
    // we have the budget for that before we compute the exact index of the
    // empty slot. Without the growth budget we'll have to completely rehash and
    // so we can just bail here.
    if (LLVM_UNLIKELY(growth_budget_ == 0)) {
      return {GrowAndInsert(hash, key_context), true};
    }

    --growth_budget_;
    CARBON_DCHECK(growth_budget() >= 0,
                  "Growth budget shouldn't have gone negative!");
    return return_insert_at_index(group_index + empty_match.index());
  }

  CARBON_FATAL(
      "We should never finish probing without finding the entry or an empty "
      "slot.");
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
[[clang::noinline]] auto
BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::GrowToAllocSizeImpl(
    ssize_t target_alloc_size, KeyContextT key_context) -> void {
  CARBON_CHECK(llvm::isPowerOf2_64(target_alloc_size));
  if (target_alloc_size <= alloc_size()) {
    return;
  }

  // If this is the next alloc size, we can used our optimized growth strategy.
  if (target_alloc_size == ComputeNextAllocSize(alloc_size())) {
    GrowToNextAllocSize(key_context);
    return;
  }

  // Create locals for the old state of the table.
  ssize_t old_size = alloc_size();
  CARBON_DCHECK(old_size > 0);
  bool old_small = is_small();
  Storage* old_storage = storage();
  uint8_t* old_metadata = metadata();
  EntryT* old_entries = entries();

  // Configure for the new size and allocate the new storage.
  alloc_size() = target_alloc_size;
  storage() = Allocate(target_alloc_size);
  std::memset(metadata(), 0, target_alloc_size);
  growth_budget_ = GrowthThresholdForAllocSize(target_alloc_size);

  // Just re-insert all the entries. As we're more than doubling the table size,
  // we don't bother with fancy optimizations here. Even using `memcpy` for the
  // entries seems unlikely to be a significant win given how sparse the
  // insertions will end up being.
  ssize_t count = 0;
  for (ssize_t group_index = 0; group_index < old_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(old_metadata, group_index);
    auto present_matched_range = g.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      ++count;
      ssize_t index = group_index + byte_index;
      HashCode hash =
          key_context.HashKey(old_entries[index].key(), ComputeSeed());
      EntryT* new_entry = InsertIntoEmpty(hash);
      new_entry->MoveFrom(std::move(old_entries[index]));
    }
  }
  growth_budget_ -= count;

  if (!old_small) {
    // Old isn't a small buffer, so we need to deallocate it.
    Deallocate(old_storage, old_size);
  }
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::GrowForInsertCountImpl(
    ssize_t count, KeyContextT key_context) -> void {
  if (count < growth_budget_) {
    // Already space for the needed growth.
    return;
  }

  // Currently, we don't account for any tombstones marking deleted elements,
  // and just conservatively ensure the growth will create adequate growth
  // budget for insertions. We could make this more precise by instead walking
  // the table and only counting present slots, as once we grow we'll be able to
  // reclaim all of the deleted slots. But this adds complexity and it isn't
  // clear this is necessary so we do the simpler conservative thing.
  ssize_t used_budget =
      GrowthThresholdForAllocSize(alloc_size()) - growth_budget_;
  ssize_t budget_needed = used_budget + count;
  ssize_t space_needed = budget_needed + (budget_needed / 7);
  ssize_t target_alloc_size = llvm::NextPowerOf2(space_needed);
  CARBON_CHECK(GrowthThresholdForAllocSize(target_alloc_size) >
               (budget_needed));
  GrowToAllocSizeImpl(target_alloc_size, key_context);
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
template <typename LookupKeyT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::EraseImpl(
    LookupKeyT lookup_key, KeyContextT key_context) -> bool {
  EntryT* entry = view_impl_.LookupEntry(lookup_key, key_context);
  if (!entry) {
    return false;
  }

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the slot's
  // metadata as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();
  ssize_t index = entry - local_entries;
  ssize_t group_index = index & ~GroupMask;
  auto g = MetadataGroup::Load(local_metadata, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    local_metadata[index] = MetadataGroup::Empty;
    ++growth_budget_;
  } else {
    local_metadata[index] = MetadataGroup::Deleted;
  }

  if constexpr (!EntryT::IsTriviallyDestructible) {
    entry->Destroy();
  }

  return true;
}

template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::ClearImpl() -> void {
  view_impl_.ForEachEntry(
      [](EntryT& entry) {
        if constexpr (!EntryT::IsTriviallyDestructible) {
          entry.Destroy();
        }
      },
      [](uint8_t* metadata_group) {
        // Clear the group.
        std::memset(metadata_group, 0, GroupSize);
      });
  growth_budget_ = GrowthThresholdForAllocSize(alloc_size());
}

// Allocates the appropriate memory layout for a table of the given
// `alloc_size`, with space both for the metadata array and entries.
//
// The returned pointer *must* be deallocated by calling the below `Deallocate`
// function with the same `alloc_size` as used here.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::Allocate(
    ssize_t alloc_size) -> Storage* {
  return reinterpret_cast<Storage*>(__builtin_operator_new(
      ViewImplT::AllocByteSize(alloc_size),
      static_cast<std::align_val_t>(Alignment), std::nothrow_t()));
}

// Deallocates a table's storage that was allocated with the `Allocate`
// function.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::Deallocate(
    Storage* storage, ssize_t alloc_size) -> void {
  ssize_t allocated_size = ViewImplT::AllocByteSize(alloc_size);
  // We don't need the size, but make sure it always compiles.
  static_cast<void>(allocated_size);
  __builtin_operator_delete(storage,
#if __cpp_sized_deallocation
                            allocated_size,
#endif
                            static_cast<std::align_val_t>(Alignment));
}

// Construct a table using the provided small storage if `small_alloc_size_` is
// non-zero. If `small_alloc_size_` is zero, then `small_storage` won't be used
// and can be null. Regardless, after this the storage pointer is non-null and
// the size is non-zero so that we can directly begin inserting or querying the
// table.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::Construct(
    Storage* small_storage) -> void {
  if (small_alloc_size_ > 0) {
    alloc_size() = small_alloc_size_;
    storage() = small_storage;
  } else {
    // Directly allocate the initial buffer so that the hashtable is never in
    // an empty state.
    alloc_size() = MinAllocatedSize;
    storage() = Allocate(MinAllocatedSize);
  }
  std::memset(metadata(), 0, alloc_size());
  growth_budget_ = GrowthThresholdForAllocSize(alloc_size());
}

// Destroy the current table, releasing any memory used.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::Destroy() -> void {
  // Check for a moved-from state and don't do anything. Only a moved-from table
  // has a zero size.
  if (alloc_size() == 0) {
    return;
  }

  // Destroy all the entries.
  if constexpr (!EntryT::IsTriviallyDestructible) {
    view_impl_.ForEachEntry([](EntryT& entry) { entry.Destroy(); },
                            [](auto...) {});
  }

  // If small, nothing to deallocate.
  if (is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  Deallocate(storage(), alloc_size());
}

// Copy all of the slots over from another table that is exactly the same
// allocation size.
//
// This requires the current table to already have storage allocated and set up
// but not initialized (or already cleared). It directly overwrites the storage
// allocation of the table to match the incoming argument.
//
// Despite being used in construction, this shouldn't be called for a moved-from
// `arg` -- in practice it is better for callers to handle this when setting up
// storage.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::CopySlotsFrom(
    const BaseImpl& arg) -> void {
  CARBON_DCHECK(alloc_size() == arg.alloc_size());
  ssize_t local_size = alloc_size();

  // Preserve which slot every entry is in, including tombstones in the
  // metadata, in order to copy into the new table's storage without rehashing
  // all of the keys. This is especially important as we don't have an easy way
  // to access the key context needed for rehashing here.
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();
  const uint8_t* local_arg_metadata = arg.metadata();
  const EntryT* local_arg_entries = arg.entries();
  memcpy(local_metadata, local_arg_metadata, local_size);

  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = MetadataGroup::Load(local_arg_metadata, group_index);
    for (ssize_t byte_index : g.MatchPresent()) {
      local_entries[group_index + byte_index].CopyFrom(
          local_arg_entries[group_index + byte_index]);
    }
  }
}

// Move from another table to this one.
//
// Note that the `small_storage` is *this* table's small storage pointer,
// provided from the `TableImpl` to this `BaseImpl` method as an argument.
//
// Requires the table to have size and growth already set up but otherwise the
// the table has not yet been initialized. Notably, storage should either not
// yet be constructed or already destroyed. It both sets up the storage and
// handles any moving slots needed.
//
// Note that because this is used in construction it needs to handle a
// moved-from `arg`.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::MoveFrom(
    BaseImpl&& arg, Storage* small_storage) -> void {
  ssize_t local_size = alloc_size();
  CARBON_DCHECK(local_size == arg.alloc_size());
  // If `arg` is moved-from, skip the rest as the local size is all we need.
  if (local_size == 0) {
    return;
  }

  if (arg.is_small()) {
    CARBON_DCHECK(local_size == small_alloc_size_);
    this->storage() = small_storage;

    // For small tables, we have to move the entries as we can't move the tables
    // themselves. We do this preserving their slots and even tombstones to
    // avoid rehashing.
    uint8_t* local_metadata = this->metadata();
    EntryT* local_entries = this->entries();
    uint8_t* local_arg_metadata = arg.metadata();
    EntryT* local_arg_entries = arg.entries();
    memcpy(local_metadata, local_arg_metadata, local_size);
    if (EntryT::IsTriviallyRelocatable) {
      memcpy(local_entries, local_arg_entries, local_size * sizeof(EntryT));
    } else {
      for (ssize_t group_index = 0; group_index < local_size;
           group_index += GroupSize) {
        auto g = MetadataGroup::Load(local_arg_metadata, group_index);
        for (ssize_t byte_index : g.MatchPresent()) {
          local_entries[group_index + byte_index].MoveFrom(
              std::move(local_arg_entries[group_index + byte_index]));
        }
      }
    }
  } else {
    // Just point to the allocated storage.
    storage() = arg.storage();
  }

  // Finally, put the incoming table into a moved-from state.
  arg.alloc_size() = 0;
  // Replace the pointer with null to ease debugging.
  arg.storage() = nullptr;
}

// Optimized routine to insert a key into a table when that key *definitely*
// isn't present in the table and the table *definitely* has a viable empty slot
// (and growth space) to insert into before any deleted slots. When both of
// these are true, typically just after growth, we can dramatically simplify the
// insert position search.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::InsertIntoEmpty(
    HashCode hash) -> EntryT* {
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t* local_metadata = metadata();
  EntryT* local_entries = entries();

  for (ProbeSequence s(hash_index, alloc_size());; s.Next()) {
    ssize_t group_index = s.index();
    auto g = MetadataGroup::Load(local_metadata, group_index);

    if (auto empty_match = g.MatchEmpty()) {
      ssize_t index = group_index + empty_match.index();
      local_metadata[index] = tag | MetadataGroup::PresentMask;
      return &local_entries[index];
    }

    // Otherwise we continue probing.
  }
}

// Apply our doubling growth strategy and (re-)check invariants around table
// size.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::ComputeNextAllocSize(
    ssize_t old_alloc_size) -> ssize_t {
  CARBON_DCHECK(llvm::isPowerOf2_64(old_alloc_size),
                "Expected a power of two!");
  ssize_t new_alloc_size;
  bool overflow = __builtin_mul_overflow(old_alloc_size, 2, &new_alloc_size);
  CARBON_CHECK(!overflow, "Computing the new size overflowed `ssize_t`!");
  return new_alloc_size;
}

// Compute the growth threshold for a given size.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT,
              InputKeyContextT>::GrowthThresholdForAllocSize(ssize_t alloc_size)
    -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return alloc_size - alloc_size / 8;
}

// Optimized routine for growing to the next alloc size.
//
// A particularly common and important-to-optimize path is growing to the next
// alloc size, which will always be a doubling of the allocated size. This
// allows an important optimization -- we're adding exactly one more high bit to
// the hash-computed index for each entry. This in turn means we can classify
// every entry in the table into three cases:
//
// 1) The new high bit is zero, the entry is at the same index in the new
//    table as the old.
//
// 2) The new high bit is one, the entry is at the old index plus the old
//    size.
//
// 3) The entry's current index doesn't match the initial hash index because
//    it required some amount of probing to find an empty slot.
//
// The design of the hash table tries to minimize how many entries fall into
// case (3), so we expect the vast majority of entries to be in (1) or (2). This
// lets us model growth notionally as copying the hashtable twice into the lower
// and higher halves of the new allocation, clearing out the now-empty slots
// (from both deleted entries and entries in the other half of the table after
// growth), and inserting any probed elements. That model in turn is much more
// efficient than re-inserting all of the elements as it avoids the unnecessary
// parts of insertion and avoids interleaving random accesses for the probed
// elements. But most importantly, for trivially relocatable types it allows us
// to use `memcpy` rather than moving the elements individually.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
auto BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::GrowToNextAllocSize(
    KeyContextT key_context) -> void {
  // We collect the probed elements in a small vector for re-insertion. It is
  // tempting to reuse the already allocated storage, but doing so appears to
  // be a (very slight) performance regression. These are relatively rare and
  // storing them into the existing storage creates stores to the same regions
  // of memory we're reading. Moreover, it requires moving both the key and the
  // value twice, and doing the `memcpy` widening for relocatable types before
  // the group walk rather than after the group walk. In practice, between the
  // statistical rareness and using a large small size buffer here on the stack,
  // we can handle this most efficiently with temporary, additional storage.
  llvm::SmallVector<std::pair<ssize_t, HashCode>, 128> probed_indices;

  // Create locals for the old state of the table.
  ssize_t old_size = alloc_size();
  CARBON_DCHECK(old_size > 0);

  bool old_small = is_small();
  Storage* old_storage = storage();
  uint8_t* old_metadata = metadata();
  EntryT* old_entries = entries();

#ifndef NDEBUG
  // Count how many of the old table slots will end up being empty after we grow
  // the table. This is both the currently empty slots, but also the deleted
  // slots because we clear them to empty and re-insert everything that had any
  // probing.
  ssize_t debug_empty_count =
      llvm::count(llvm::ArrayRef(old_metadata, old_size), MetadataGroup::Empty);
  ssize_t debug_deleted_count = llvm::count(
      llvm::ArrayRef(old_metadata, old_size), MetadataGroup::Deleted);
  CARBON_DCHECK(
      debug_empty_count >= (old_size - GrowthThresholdForAllocSize(old_size)),
      "debug_empty_count: {0}, debug_deleted_count: {1}, size: {2}",
      debug_empty_count, debug_deleted_count, old_size);
#endif

  // Configure for the new size and allocate the new storage.
  ssize_t new_size = ComputeNextAllocSize(old_size);
  alloc_size() = new_size;
  storage() = Allocate(new_size);
  growth_budget_ = GrowthThresholdForAllocSize(new_size);

  // Now extract the new components of the table.
  uint8_t* new_metadata = metadata();
  EntryT* new_entries = entries();

  // Walk the metadata groups, clearing deleted to empty, duplicating the
  // metadata for the low and high halves, and updating it based on where each
  // entry will go in the new table. The updated metadata group is written to
  // the new table, and for non-trivially relocatable entry types, the entry is
  // also moved to its new location.
  ssize_t count = 0;
  for (ssize_t group_index = 0; group_index < old_size;
       group_index += GroupSize) {
    auto low_g = MetadataGroup::Load(old_metadata, group_index);
    // Make sure to match present elements first to enable pipelining with
    // clearing.
    auto present_matched_range = low_g.MatchPresent();
    low_g.ClearDeleted();
    MetadataGroup high_g;
    if constexpr (MetadataGroup::FastByteClear) {
      // When we have a fast byte clear, we can update the metadata for the
      // growth in-register and store at the end.
      high_g = low_g;
    } else {
      // If we don't have a fast byte clear, we can store the metadata group
      // eagerly here and overwrite bytes with a byte store below instead of
      // clearing the byte in-register.
      low_g.Store(new_metadata, group_index);
      low_g.Store(new_metadata, group_index | old_size);
    }
    for (ssize_t byte_index : present_matched_range) {
      ++count;
      ssize_t old_index = group_index + byte_index;
      if constexpr (!MetadataGroup::FastByteClear) {
        CARBON_DCHECK(new_metadata[old_index] == old_metadata[old_index]);
        CARBON_DCHECK(new_metadata[old_index | old_size] ==
                      old_metadata[old_index]);
      }
      HashCode hash =
          key_context.HashKey(old_entries[old_index].key(), ComputeSeed());
      ssize_t old_hash_index = hash.ExtractIndexAndTag<7>().first &
                               ComputeProbeMaskFromSize(old_size);
      if (LLVM_UNLIKELY(old_hash_index != group_index)) {
        probed_indices.push_back({old_index, hash});
        if constexpr (MetadataGroup::FastByteClear) {
          low_g.ClearByte(byte_index);
          high_g.ClearByte(byte_index);
        } else {
          new_metadata[old_index] = MetadataGroup::Empty;
          new_metadata[old_index | old_size] = MetadataGroup::Empty;
        }
        continue;
      }
      ssize_t new_index = hash.ExtractIndexAndTag<7>().first &
                          ComputeProbeMaskFromSize(new_size);
      CARBON_DCHECK(new_index == old_hash_index ||
                    new_index == (old_hash_index | old_size));
      // Toggle the newly added bit of the index to get to the other possible
      // target index.
      if constexpr (MetadataGroup::FastByteClear) {
        (new_index == old_hash_index ? high_g : low_g).ClearByte(byte_index);
        new_index += byte_index;
      } else {
        new_index += byte_index;
        new_metadata[new_index ^ old_size] = MetadataGroup::Empty;
      }

      // If we need to explicitly move (and destroy) the key or value, do so
      // here where we already know its target.
      if constexpr (!EntryT::IsTriviallyRelocatable) {
        new_entries[new_index].MoveFrom(std::move(old_entries[old_index]));
      }
    }
    if constexpr (MetadataGroup::FastByteClear) {
      low_g.Store(new_metadata, group_index);
      high_g.Store(new_metadata, (group_index | old_size));
    }
  }
  CARBON_DCHECK((count - static_cast<ssize_t>(probed_indices.size())) ==
                (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                        MetadataGroup::Empty)));
#ifndef NDEBUG
  CARBON_DCHECK((debug_empty_count + debug_deleted_count) ==
                (old_size - count));
  CARBON_DCHECK(llvm::count(llvm::ArrayRef(new_metadata, new_size),
                            MetadataGroup::Empty) ==
                debug_empty_count + debug_deleted_count +
                    static_cast<ssize_t>(probed_indices.size()) + old_size);
#endif

  // If the keys or values are trivially relocatable, we do a bulk memcpy of
  // them into place. This will copy them into both possible locations, which is
  // fine. One will be empty and clobbered if reused or ignored. The other will
  // be the one used. This might seem like it needs it to be valid for us to
  // create two copies, but it doesn't. This produces the exact same storage as
  // copying the storage into the wrong location first, and then again into the
  // correct location. Only one is live and only one is destroyed.
  if constexpr (EntryT::IsTriviallyRelocatable) {
    memcpy(new_entries, old_entries, old_size * sizeof(EntryT));
    memcpy(new_entries + old_size, old_entries, old_size * sizeof(EntryT));
  }

  // We then need to do a normal insertion for anything that was probed before
  // growth, but we know we'll find an empty slot, so leverage that.
  for (auto [old_index, hash] : probed_indices) {
    EntryT* new_entry = InsertIntoEmpty(hash);
    new_entry->MoveFrom(std::move(old_entries[old_index]));
  }
  CARBON_DCHECK(count ==
                (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                        MetadataGroup::Empty)));
  growth_budget_ -= count;
  CARBON_DCHECK(growth_budget_ ==
                (GrowthThresholdForAllocSize(new_size) -
                 (new_size - llvm::count(llvm::ArrayRef(new_metadata, new_size),
                                         MetadataGroup::Empty))));
  CARBON_DCHECK(growth_budget_ > 0 &&
                "Must still have a growth budget after rehash!");

  if (!old_small) {
    // Old isn't a small buffer, so we need to deallocate it.
    Deallocate(old_storage, old_size);
  }
}

// Grow the hashtable to create space and then insert into it. Returns the
// selected insertion entry. Never returns null. In addition to growing and
// selecting the insertion entry, this routine updates the metadata array so
// that this function can be directly called and the result returned from
// `InsertImpl`.
template <typename InputKeyT, typename InputValueT, typename InputKeyContextT>
[[clang::noinline]] auto
BaseImpl<InputKeyT, InputValueT, InputKeyContextT>::GrowAndInsert(
    HashCode hash, KeyContextT key_context) -> EntryT* {
  GrowToNextAllocSize(key_context);

  // And insert the lookup_key into an index in the newly grown map and return
  // that index for use.
  --growth_budget_;
  return InsertIntoEmpty(hash);
}

template <typename InputBaseT, ssize_t SmallSize>
TableImpl<InputBaseT, SmallSize>::TableImpl(const TableImpl& arg)
    : BaseT(arg.alloc_size(), arg.growth_budget_, SmallSize) {
  // Check for completely broken objects. These invariants should be true even
  // in a moved-from state.
  CARBON_DCHECK(arg.alloc_size() == 0 || !arg.is_small() ||
                arg.alloc_size() == SmallSize);
  CARBON_DCHECK(arg.small_alloc_size_ == SmallSize);
  CARBON_DCHECK(this->small_alloc_size_ == SmallSize);

  if (this->alloc_size() != 0) {
    SetUpStorage();
    this->CopySlotsFrom(arg);
  }
}

template <typename InputBaseT, ssize_t SmallSize>
auto TableImpl<InputBaseT, SmallSize>::operator=(const TableImpl& arg)
    -> TableImpl& {
  // Check for completely broken objects. These invariants should be true even
  // in a moved-from state.
  CARBON_DCHECK(arg.alloc_size() == 0 || !arg.is_small() ||
                arg.alloc_size() == SmallSize);
  CARBON_DCHECK(arg.small_alloc_size_ == SmallSize);
  CARBON_DCHECK(this->small_alloc_size_ == SmallSize);

  // We have to end up with an allocation size exactly equivalent to the
  // incoming argument to avoid re-hashing every entry in the table, which isn't
  // possible without key context.
  if (arg.alloc_size() == this->alloc_size()) {
    // No effective way for self-assignment to fall out of an efficient
    // implementation so detect and bypass here. Similarly, if both are in a
    // moved-from state, there is nothing to do.
    if (&arg == this || this->alloc_size() == 0) {
      return *this;
    }
    CARBON_DCHECK(arg.storage() != this->storage());
    if constexpr (!EntryT::IsTriviallyDestructible) {
      this->view_impl_.ForEachEntry([](EntryT& entry) { entry.Destroy(); },
                                    [](auto...) {});
    }
  } else {
    // The sizes don't match so destroy everything and re-setup the table
    // storage.
    this->Destroy();
    this->alloc_size() = arg.alloc_size();
    // If `arg` is moved-from, we've clear out our elements and put ourselves
    // into a moved-from state. We're done.
    if (this->alloc_size() == 0) {
      return *this;
    }
    SetUpStorage();
  }
  this->growth_budget_ = arg.growth_budget_;
  this->CopySlotsFrom(arg);
  return *this;
}

// Puts the incoming table into a moved-from state that can be destroyed or
// re-initialized but must not be used otherwise.
template <typename InputBaseT, ssize_t SmallSize>
TableImpl<InputBaseT, SmallSize>::TableImpl(TableImpl&& arg) noexcept
    : BaseT(arg.alloc_size(), arg.growth_budget_, SmallSize) {
  // Check for completely broken objects. These invariants should be true even
  // in a moved-from state.
  CARBON_DCHECK(arg.alloc_size() == 0 || !arg.is_small() ||
                arg.alloc_size() == SmallSize);
  CARBON_DCHECK(arg.small_alloc_size_ == SmallSize);
  CARBON_DCHECK(this->small_alloc_size_ == SmallSize);
  this->MoveFrom(std::move(arg), small_storage());
}

template <typename InputBaseT, ssize_t SmallSize>
auto TableImpl<InputBaseT, SmallSize>::operator=(TableImpl&& arg) noexcept
    -> TableImpl& {
  // Check for completely broken objects. These invariants should be true even
  // in a moved-from state.
  CARBON_DCHECK(arg.alloc_size() == 0 || !arg.is_small() ||
                arg.alloc_size() == SmallSize);
  CARBON_DCHECK(arg.small_alloc_size_ == SmallSize);
  CARBON_DCHECK(this->small_alloc_size_ == SmallSize);

  // Destroy and deallocate our table.
  this->Destroy();

  // Defend against self-move by zeroing the size here before we start moving
  // out of `arg`.
  this->alloc_size() = 0;

  // Setup to match argument and then finish the move.
  this->alloc_size() = arg.alloc_size();
  this->growth_budget_ = arg.growth_budget_;
  this->MoveFrom(std::move(arg), small_storage());
  return *this;
}

template <typename InputBaseT, ssize_t SmallSize>
TableImpl<InputBaseT, SmallSize>::~TableImpl() {
  this->Destroy();
}

// Reset a table to its original state, including releasing any allocated
// memory.
template <typename InputBaseT, ssize_t SmallSize>
auto TableImpl<InputBaseT, SmallSize>::ResetImpl() -> void {
  this->Destroy();

  // Re-initialize the whole thing.
  CARBON_DCHECK(this->small_alloc_size() == SmallSize);
  this->Construct(small_storage());
}

template <typename InputBaseT, ssize_t SmallSize>
auto TableImpl<InputBaseT, SmallSize>::small_storage() const -> Storage* {
  if constexpr (SmallSize > 0) {
    // Do a bunch of validation of the small size to establish our invariants
    // when we know we have a non-zero small size.
    static_assert(llvm::isPowerOf2_64(SmallSize),
                  "SmallSize must be a power of two for a hashed buffer!");
    static_assert(
        SmallSize >= MaxGroupSize,
        "We require all small sizes to multiples of the largest group "
        "size supported to ensure it can be used portably.  ");
    static_assert(
        (SmallSize % MaxGroupSize) == 0,
        "Small size must be a multiple of the max group size supported "
        "so that we can allocate a whole number of groups.");
    // Implied by the max asserts above.
    static_assert(SmallSize >= GroupSize);
    static_assert((SmallSize % GroupSize) == 0);

    static_assert(SmallSize >= alignof(StorageEntry<KeyT, ValueT>),
                  "Requested a small size that would require padding between "
                  "metadata bytes and correctly aligned key and value types. "
                  "Either a larger small size or a zero small size and heap "
                  "allocation are required for this key and value type.");

    static_assert(offsetof(SmallStorage, entries) == SmallSize,
                  "Offset to entries in small size storage doesn't match "
                  "computed offset!");

    return &small_storage_;
  } else {
    static_assert(
        sizeof(TableImpl) == sizeof(BaseT),
        "Empty small storage caused a size difference and wasted space!");

    return nullptr;
  }
}

// Helper to set up the storage of a table when a specific size has already been
// set up. If possible, uses any small storage, otherwise allocates.
template <typename InputBaseT, ssize_t SmallSize>
auto TableImpl<InputBaseT, SmallSize>::SetUpStorage() -> void {
  CARBON_DCHECK(this->small_alloc_size() == SmallSize);
  ssize_t local_size = this->alloc_size();
  CARBON_DCHECK(local_size != 0);
  if (local_size == SmallSize) {
    this->storage() = small_storage();
  } else {
    this->storage() = BaseT::Allocate(local_size);
  }
}

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_H_
