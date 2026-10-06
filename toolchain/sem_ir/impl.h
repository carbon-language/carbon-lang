// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
#define CARBON_TOOLCHAIN_SEM_IR_IMPL_H_

#include <utility>

#include "common/map.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/declared_facet_type.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct ImplFields {
  // The following members are set at the start of the impl declaration.

  // The name scope containing the impl (as opposed to the scope _of_ the impl).
  // Note that this is None for an imported impl.
  InstId parent_scope_inst_id;

  // Whether the impl declaration is marked `final`. This is false for impls in
  // a `final match_first`. The `match_first_is_final` flag is used instead for
  // that case.
  bool is_final;

  // The following members are set at the start of the impl declaration, and may
  // be modified at the start of redeclarations.

  // The `MatchFirstDecl` of the `match_first` block that the impl is associated
  // with.
  InstId match_first_id = SemIR::InstId::None;
  // The location of the `ImplDecl` that was associated with the
  // `match_first_id`. Used for diagnostics, and not imported.
  LocId decl_loc_in_match_first = SemIR::LocId::None;
  // The position of the impl in its associated `match_first` block.
  int match_first_position = 0;
  // Whether the `match_first` block was modified as `final`.
  bool match_first_is_final = false;

  // The following members always have values and do not change.

  // The type for which the impl is implementing a constraint.
  TypeInstId self_id;
  // The constraint that the impl implements, which only contains extended
  // interfaces or named constraints. Other constraints such as rewrites are not
  // preserved in this instruction, as they contain `.Self` and thus only make
  // sense internally.
  TypeInstId constraint_id;

  // The single interface to implement from `constraint_id`. This comes from the
  // IdentifiedFacetType of the constraint, so any `.Self` are replaced by the
  // impl's self type.
  //
  // May be `None` if the impl's constraint is not valid.
  SpecificInterface interface;
  // An `ImplSelfWitness` instruction, which contains the SpecificInterface
  // inside it. If the impl is generic, the constant value of the instruction
  // has the impl's specific applied to it, so that GetConstantValueInSpecific
  // can be used to get the `interface` being implemented after deducing the
  // impl's generic arguments.
  InstId interface_inst_id;

  // The witness for the impl. This can be `BuiltinErrorInst` or an import
  // reference. Note that the entries in the witness are updated at the end of
  // the impl definition.
  InstId witness_id = InstId::None;
  // A block for instructions that make up the impl's witness so that they can
  // be formatted as part of the impl.
  InstBlockId witness_block_id = InstBlockId::None;

  // The following members are set at the `{` of the impl definition.

  // The impl scope.
  NameScopeId scope_id = NameScopeId::None;
  // The first block of the impl body.
  // TODO: Handle control flow in the impl body, such as if-expressions.
  InstBlockId body_block_id = InstBlockId::None;

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

  explicit ImplStore(File& sem_ir);

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

  auto GetRawIndex(ImplId id) const -> int32_t {
    return values_.GetRawIndex(id);
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "lookup_"), lookup_);
  }

  auto values() const [[clang::lifetimebound]]
  -> ValueStore<ImplId, Impl, Tag<CheckIRId>>::Range {
    return values_.values();
  }
  auto size() const -> size_t { return values_.size(); }
  auto enumerate() const [[clang::lifetimebound]] -> auto {
    return values_.enumerate();
  }

 private:
  File& sem_ir_;
  ValueStore<ImplId, Impl, Tag<CheckIRId>> values_;
  Map<std::pair<ConstantId, SpecificInterface>, ImplOrLookupBucketId> lookup_;
  // Buckets with at least 2 entries, which will be rare; see LookupBucketRef.
  llvm::SmallVector<llvm::SmallVector<ImplId, 2>> lookup_buckets_;
};

inline constexpr ImplStore::ImplOrLookupBucketId
    ImplStore::ImplOrLookupBucketId::None(NoneIndex);

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::ImplId, SemIR::Impl,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_IMPL_H_
