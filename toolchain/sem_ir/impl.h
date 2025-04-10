// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
#define CARBON_TOOLCHAIN_SEM_IR_IMPL_H_

#include "common/map.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct ImplFields {
  // This following members always have values and do not change.

  // The type for which the impl is implementing a constraint.
  InstId self_id;
  // The constraint that the impl implements.
  InstId constraint_id;

  // The single interface to implement from `constraint_id`.
  // The members are `None` if `constraint_id` isn't complete or doesn't
  // correspond to a single interface.
  SemIR::SpecificInterface interface;

  // The witness for the impl. This can be `BuiltinErrorInst` or an import
  // reference. Note that the entries in the witness are updated at the end of
  // the impl definition.
  InstId witness_id = InstId::None;

  // The following members are set at the `{` of the impl definition.

  // The impl scope.
  NameScopeId scope_id = NameScopeId::None;
  // The first block of the impl body.
  // TODO: Handle control flow in the impl body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::None;

  // Whether the impl declaration is marked `final`.
  bool is_final;

  // The following members are set at the `}` of the impl definition.
  bool defined = false;
};

// An implementation of a constraint. See EntityWithParamsBase regarding the
// inheritance here.
struct Impl : public EntityWithParamsBase,
              public ImplFields,
              public Printable<Impl> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{self: " << self_id << ", constraint: " << constraint_id
        << ", witness: " << witness_id << "}";
  }

  // This is false until we reach the `}` of the impl definition.
  auto is_complete() const -> bool { return defined; }

  // Determines whether this impl's definition has begun but not yet ended.
  auto is_being_defined() const -> bool {
    return has_definition_started() && !is_complete();
  }
};

// A collection of `Impl`s, which can be accessed by the self type and
// constraint implemented.
class ImplStore {
 private:
  // An ID of either a single impl or a lookup bucket.
  class ImplOrLookupBucketId : public IdBase<ImplOrLookupBucketId> {
   public:
    static constexpr llvm::StringLiteral Label = "impl_or_lookup_bucket";

    // An ID with no value, corresponding to to ImplId::None.
    static const ImplOrLookupBucketId None;

    static auto ForImplId(ImplId impl_id) -> ImplOrLookupBucketId {
      return ImplOrLookupBucketId(impl_id.index);
    }

    static auto ForBucket(int bucket) -> ImplOrLookupBucketId {
      return ImplOrLookupBucketId(ImplId::NoneIndex - bucket - 1);
    }

    // Returns whether this ID represents a bucket index, rather than an ImplId.
    // `None` is not a bucket index.
    auto is_bucket() const { return index < ImplId::NoneIndex; }

    // Returns the bucket index represented by this ID. Requires is_bucket().
    auto bucket() const -> int {
      CARBON_CHECK(is_bucket());
      return ImplId::NoneIndex - index - 1;
    }

    // Returns the ImplId index represented by this ID. Requires !is_bucket().
    auto impl_id() const -> ImplId {
      CARBON_CHECK(!is_bucket());
      return ImplId(index);
    }

   private:
    explicit constexpr ImplOrLookupBucketId(int index) : IdBase(index) {}
  };

 public:
  // A reference to an impl lookup bucket. This represents a list of impls with
  // the same self and constraint type.
  //
  // The bucket is held indirectly as an `ImplOrLookupBucketId`, in one of three
  // states:
  //
  //   - `ImplId::None` represents an empty bucket.
  //   - An `ImplId` value represents a bucket with exactly one impl. This is
  //     expected to be by far the most common case.
  //   - A lookup bucket index represents an index within the `ImplStore`'s
  //     array of variable-sized lookup buckets.
  class LookupBucketRef {
   public:
    LookupBucketRef(ImplStore& store, ImplOrLookupBucketId& id)
        : store_(&store), id_(&id), single_id_storage_(ImplId::None) {
      if (!id.is_bucket()) {
        single_id_storage_ = id.impl_id();
      }
    }

    auto begin() const -> const ImplId* {
      if (id_->is_bucket()) {
        return store_->lookup_buckets_[id_->bucket()].begin();
      }
      return &single_id_storage_;
    }

    auto end() const -> const ImplId* {
      if (id_->is_bucket()) {
        return store_->lookup_buckets_[id_->bucket()].end();
      }
      return &single_id_storage_ + (id_->has_value() ? 1 : 0);
    }

    // Adds an impl to this lookup bucket. Only impls from the current file and
    // its API file should be added in this way. Impls from other files do not
    // need to be found by impl redeclaration lookup so should not be added.
    auto push_back(ImplId impl_id) -> void {
      if (!id_->has_value()) {
        *id_ = ImplOrLookupBucketId::ForImplId(impl_id);
        single_id_storage_ = impl_id;
      } else if (!id_->is_bucket()) {
        auto first_id = id_->impl_id();
        *id_ = ImplOrLookupBucketId::ForBucket(store_->lookup_buckets_.size());
        store_->lookup_buckets_.push_back({first_id, impl_id});
      } else {
        store_->lookup_buckets_[id_->bucket()].push_back(impl_id);
      }
    }

   private:
    ImplStore* store_;
    ImplOrLookupBucketId* id_;
    // Storage for a single ImplId. Used to support iteration over the contents
    // of the bucket when it contains a single ImplId.
    ImplId single_id_storage_;
  };

  explicit ImplStore(File& sem_ir) : sem_ir_(sem_ir) {}

  // Returns a reference to the lookup bucket containing the list of impls with
  // this self type and constraint, or adds a new bucket if this is the first
  // time we've seen an impl of this kind. The lookup bucket reference remains
  // valid until this function is called again.
  auto GetOrAddLookupBucket(const Impl& impl) -> LookupBucketRef;

  // Adds the specified impl to the store. Does not add it to impl lookup.
  auto Add(Impl impl) -> ImplId { return values_.Add(impl); }

  // Returns a mutable value for an ID.
  auto Get(ImplId id) -> Impl& { return values_.Get(id); }

  // Returns the value for an ID.
  auto Get(ImplId id) const -> const Impl& { return values_.Get(id); }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "lookup_"), lookup_);
  }

  auto array_ref() const -> llvm::ArrayRef<Impl> { return values_.array_ref(); }
  auto size() const -> size_t { return values_.size(); }
  auto enumerate() const -> auto { return values_.enumerate(); }

 private:
  File& sem_ir_;
  ValueStore<ImplId> values_;
  Map<std::tuple<InstId, InterfaceId, SpecificId>, ImplOrLookupBucketId>
      lookup_;
  // Buckets with at least 2 entries, which will be rare; see LookupBucketRef.
  llvm::SmallVector<llvm::SmallVector<ImplId, 2>> lookup_buckets_;
};

constexpr inline ImplStore::ImplOrLookupBucketId
    ImplStore::ImplOrLookupBucketId::None(NoneIndex);

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
