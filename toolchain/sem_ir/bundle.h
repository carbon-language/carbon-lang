// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BUNDLE_H_
#define CARBON_TOOLCHAIN_SEM_IR_BUNDLE_H_

#include <string>
#include <utility>

#include "common/raw_string_ostream.h"
#include "common/struct_reflection.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/base/block_value_store.h"
#include "toolchain/base/id_tag.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A value store for bundles of instruction arguments.
//
// When an inst needs to take more than two logical arguments, one of its
// physical arguments can be the ID of a "bundle" of extra arguments, which is
// stored here. The expected pattern looks like this:
//
// struct MyInst {
//   struct Args {
//     FooId arg1;
//     BarId arg2;
//     BazId arg3;
//   };
//
//   static constexpr auto Kind = ...
//
//   TypeId type_id;
//   QuuxId arg0;
//   BundleId<Args> extra_args;
// };
//
// A bundle type like `Args` must be an aggregate, and its fields must all be
// ID types, i.e. types listed in the definition of `IdKind`. `BundleId<Args>`
// itself must also be added to that list, although bundles should generally not
// have bundle IDs as members.
//
// Unlike insts, bundles do not record their own kind and the `BundleStore`
// is not guaranteed to record it either. Instead, that information is tracked
// by the user, usually in the bundle ID's static type. It may also be tracked
// dynamically with an `IdAndKind`, or discarded entirely by converting to a
// `RawBundleId`.
//
// The `BundleStore` can cache the kinds of the stored bundles for debugging
// purposes, but this must be requested separately by calling `CacheDebugKind`
// because it adds storage overhead, and it affects only the store's debug
// printing operations.
// TODO: Consider storing and populating the cache separately, and passing it
// into the debug printing operations.
//
// In rare cases, a single `RawBundleId` may correspond to multiple typed
// bundle IDs, and hence to multiple bundles with different types. However,
// this happens only with the canonical IDs of bundles that have the same
// representation (i.e. the same sequence of integer IDs), so this has no
// practical effect other than somewhat complicating the debug printing.
class BundleStore {
 public:
  // Construct a `BundleStore` that uses the given allocator, and applies the
  // given tag to bundle IDs.
  explicit BundleStore(llvm::BumpPtrAllocator& allocator, CheckIRId tag_id)
      : store_(allocator, tag_id, 0), bundle_kind_cache_(store_.GetIdTag()) {}

  // Adds a new bundle to the store, and returns its ID.
  template <typename BundleT>
  auto Add(const BundleT& bundle) -> BundleId<BundleT> {
    return BundleId<BundleT>{store_.Add(BundleToArray(bundle))};
  }

  // Returns the canonical ID of the given bundle, allocating a new one if
  // it does not already exist.
  template <typename BundleT>
  auto AddCanonical(const BundleT& bundle) -> BundleId<BundleT> {
    return BundleId<BundleT>{store_.AddCanonical(BundleToArray(bundle))};
  }

  // Returns the canonical ID of the bundle specified by `bundle_id`, allocating
  // a new canonical ID if none exists already.
  template <typename BundleT>
  auto MakeCanonical(BundleId<BundleT> bundle_id) -> BundleId<BundleT> {
    return BundleId<BundleT>{store_.MakeCanonical(bundle_id.index)};
  }

  // Returns the bundle with the given ID.
  template <typename BundleT>
  auto Get(BundleId<BundleT> bundle_id) const -> BundleT {
    using TupleType =
        decltype(StructReflection::AsTuple(std::declval<BundleT>()));
    return FromArray<TupleType>::template AsBundle<BundleT>(
        store_.Get(bundle_id));
  }

  // Returns a std::tuple of the fields of the bundle with the given ID, in
  // declaration order.
  template <typename BundleT>
  auto GetAsTuple(BundleId<BundleT> bundle_id) const
      -> decltype(StructReflection::AsTuple(std::declval<BundleT>())) {
    using TupleType =
        decltype(StructReflection::AsTuple(std::declval<BundleT>()));
    return FromArray<TupleType>::AsTuple(store_.Get(bundle_id));
  }

  // Caches additional information about `bundle_id` for debug-printing
  // purposes. This incurs some storage overhead, so it should only be called
  // before one of the methods that relies on it.
  template <typename BundleT>
  auto CacheDebugKind(BundleId<BundleT> bundle_id) const -> void {
    if (bundle_kind_cache_.size() < store_.size()) {
      bundle_kind_cache_.Resize(store_.size(), IdKindSet{});
    }
    bundle_kind_cache_.Get(bundle_id).insert(IdKind::For<BundleId<BundleT>>);
  }

  // Returns the contents of the bundle store as a YAML mapping of untyped
  // bundle IDs to bundle values. If an untyped ID corresponds to multiple
  // typed IDs, the entry for that ID will be a nested mapping with the
  // different bundles as values.
  //
  // Each bundle's fields will be depicted with the correct ID kinds if
  // CacheDebugKind was previously called for that bundle; otherwise they will
  // be depicted as AnyRawIds, and may be conflated with other bundles that have
  // the same untyped ID (and hence the same numeric field values).
  auto OutputYaml() const -> Yaml::OutputMapping;

  // Equivalent to OutputYaml, but the resulting mapping will contain only
  // the given bundle.
  auto OutputBundleYaml(RawBundleId bundle_id) const -> Yaml::OutputMapping;

  // Adds the store's memory usage to mem_usage.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // The number of stored bundles.
  auto size() const -> size_t { return store_.size(); }

 private:
  // Comparator for sets of `IdKind`.
  struct IdKindLess {
    auto operator()(IdKind lhs, IdKind rhs) const -> bool {
      return lhs.ToIndex() < rhs.ToIndex();
    }
  };
  // A set of IdKinds, for use in the debug info cache.
  using IdKindSet = llvm::SmallSet<IdKind, 1, IdKindLess>;

  // Returns the fields of the given bundle as a `std::array<AnyRawId>`.
  template <typename BundleT>
  auto BundleToArray(const BundleT& bundle) -> auto {
    static_assert(std::is_aggregate_v<BundleT>,
                  "Only aggregates are supported");
    return std::apply(
        [](auto... ids)->std::array<AnyRawId, sizeof...(ids)> {
          return {AnyRawId(ToRaw(ids))...};
        },
        StructReflection::AsTuple(bundle));
  }

  // Helper class for converting an array of `AnyRawId`s back to typed IDs.
  // TupleT must be a std::tuple of the bundle's field types, in declaration
  // order.
  template <typename TupleT>
  class FromArray;
  template <typename... Ts>
  class FromArray<std::tuple<Ts...>> {
   public:
    // Returns the field values in `array` as a bundle of type `BundleT`.
    template <typename BundleT>
    static auto AsBundle(llvm::ArrayRef<AnyRawId> array) -> BundleT {
      static_assert(std::is_aggregate_v<BundleT>,
                    "Only aggregates are supported");
      return As<BundleT>(array, std::make_index_sequence<sizeof...(Ts)>{});
    }

    // Returns the field values in `array` as a tuple of type `TupleT`.
    static auto AsTuple(llvm::ArrayRef<AnyRawId> array) -> std::tuple<Ts...> {
      return As<std::tuple<Ts...>>(array,
                                   std::make_index_sequence<sizeof...(Ts)>{});
    }

   private:
    template <typename ResultT, size_t... Is>
    static auto As(llvm::ArrayRef<AnyRawId> array,
                   std::index_sequence<Is...> /*is*/) -> ResultT {
      CARBON_CHECK(array.size() == sizeof...(Ts));
      return {FromRaw<Ts>(array[Is].index)...};
    }
  };

  // Prints a single field of a bundle.
  template <typename T>
  static auto PrintBundleField(llvm::raw_ostream& out, llvm::ListSeparator& sep,
                               size_t i, T field) -> void {
    out << sep << "arg" << i << ": " << field;
  }

  // Returns a YAML string representation of the bundle with the given ID.
  // This is overloaded for all ID types, for use with `IdAndKind::Dispatch`,
  // but should only actually be called with raw or typed bundle IDs.
  template <typename BundleT>
  auto BundleString(BundleId<BundleT> bundle_id) const -> std::string {
    static_assert(std::is_aggregate_v<BundleT>,
                  "Only aggregates are supported");
    RawStringOstream out;
    llvm::ListSeparator sep;
    size_t i = 0;
    out << "{";
    std::apply(
        [&](auto... ids) { (..., PrintBundleField(out, sep, i++, ids)); },
        GetAsTuple(bundle_id));
    out << "}";
    return out.TakeStr();
  }

  auto BundleString(RawBundleId bundle_id) const -> std::string;

  template <typename IdT>
    requires Internal::IsIdKindType<IdT>
  auto BundleString(IdT bundle_id) const -> std::string {
    CARBON_FATAL("ID {} is not a bundle ID", bundle_id);
  }

  // Adds an entry for `bundle_id` to `map`.
  auto AddYamlMapEntry(Yaml::OutputMapping::Map& map,
                       RawBundleId bundle_id) const -> void;

  // Returns a YAML mapping with an entry for each ID kind in `kinds`,
  // consisting of the string representation of the given bundle, interpreted as
  // a bundle of that kind. All entries in `kinds` must be specializations of
  // `BundleId`.
  auto YamlBundleMap(RawBundleId bundle_id, const IdKindSet& kinds) const
      -> Yaml::OutputMapping;

  // The bundles in the store, represented as blocks of `AnyRawId`s.
  //
  // TODO: Consider instead representing this as a flat array of `AnyRawId`s,
  // with the Bundle ID's index pointing to the start of the bundle in the
  // array. This would probably be more efficient, because it would be a single
  // contiguous allocation, and would avoid redundantly storing the size of
  // each block. However, it would require a different approach to debug
  // printing, which currently needs to get the bundle size from `store_`
  // when the kind cache is incomplete or unavailable.
  BlockValueStore<RawBundleId, AnyRawId, Tag<CheckIRId>> store_;

  // The cached ID kinds for the bundles in `store_`.
  //
  // TODO: Consider factoring this out as a separate class, which is populated
  // by the user and then passed into the debug-printing methods. That would
  // avoid the need for a mutable member, and reduce the risk of the cache
  // being incomplete.
  mutable ValueStore<RawBundleId, IdKindSet, Tag<CheckIRId>> bundle_kind_cache_;
};

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class BlockValueStore<SemIR::RawBundleId, SemIR::AnyRawId,
                                      Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_BUNDLE_H_
