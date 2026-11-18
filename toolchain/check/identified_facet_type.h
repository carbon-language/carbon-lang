// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IDENTIFIED_FACET_TYPE_H_
#define CARBON_TOOLCHAIN_CHECK_IDENTIFIED_FACET_TYPE_H_

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/declared_facet_type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// The ID of an identified facet type, which flattens referenced named
// constraints together into a single structure, and replaces any `.Self`
// references with a self facet.
struct IdentifiedFacetTypeId : public IdBase<IdentifiedFacetTypeId> {
  static constexpr llvm::StringLiteral Label = "identified_facet_type";

  using IdBase::IdBase;
};

class Context;

struct IdentifiedFacetTypeKey {
  SemIR::DeclaredFacetTypeId declared_facet_type_id;
  SemIR::ConstantId self_const_id;
  // Inside a named constraint, each identification of the `Self` facet type can
  // be unique, as it can be modified by each require declaration seen so far.
  // Uses -1 for identifying a facet type with a self-type from outside the
  // definition of an named constraint.
  int32_t num_require_impls = -1;

  friend auto operator==(const IdentifiedFacetTypeKey& lhs,
                         const IdentifiedFacetTypeKey& rhs) -> bool = default;
};

class SubstIdentifiedWitnessesCallbacks {
 public:
  explicit SubstIdentifiedWitnessesCallbacks(Context* context)
      : context_(context) {}

  auto context() const -> Context& { return *context_; }

  // Callback to return an InstId to use in place of the IdentifiedWitness. If
  // it returns None, the substitution will replace it with a LookupImplWitness
  // which will be evaluated to perform an impl lookup.
  virtual auto ReplaceIdentified(SemIR::IdentifiedWitness witness)
      -> SemIR::InstId = 0;

  // Callback to return an InstId to use in place of the ImplSelfWitness. If it
  // returns None, the ImplSelfWitness will be left in place.
  virtual auto ReplaceImplSelfWitness(SemIR::ImplSelfWitness witness)
      -> SemIR::InstId = 0;

 private:
  Context* context_;
};

class ConstantWithIdentifiedWitnesses {
 public:
  explicit(false) ConstantWithIdentifiedWitnesses(SemIR::ConstantId const_id)
      : const_id_(const_id) {}

  // Replaces all IdentifiedWitnesses with LookupImplWitnesses, which will
  // evaluate and perform an impl lookup. If part of an ImplWitnessAccess, the
  // result may replace the access and witness with the value of an associated
  // constant from an impl. Returns the resulting constant value.
  auto EvaluateIdentifiedWitnesses(Context& context, SemIR::LocId loc_id) const
      -> SemIR::ConstantId;
  // Replaces all IdentifiedWitnesses based on the behaviour of a callback that
  // is run as each witness is visited. Also gives the opportunity to replace
  // ImplSelfWitness in the same substitution pass.
  auto SubstIdentifiedWitnesses(
      Context& context, SemIR::LocId loc_id,
      SubstIdentifiedWitnessesCallbacks& callbacks) const -> SemIR::ConstantId;

  // Returns the constant value with the IdentifiedWitnesses present. As a
  // result, this value can't be used for constant comparisons, and does not
  // represent the final evaluated value.
  auto RawValue() const -> SemIR::ConstantId { return const_id_; }

  friend auto operator==(const ConstantWithIdentifiedWitnesses& lhs,
                         const ConstantWithIdentifiedWitnesses& rhs)
      -> bool = default;

 private:
  SemIR::ConstantId const_id_;
};

class InterfaceWithIdentifiedWitnesses {
 public:
  explicit(false)
      InterfaceWithIdentifiedWitnesses(SemIR::SpecificInterface interface)
      : interface_(interface) {}

  // Replaces all IdentifiedWitnesses with LookupImplWitnesses in the
  // interface's specific, which will evaluate and perform an impl lookup. If
  // part of an ImplWitnessAccess, the result may replace the access and witness
  // with the value of an associated constant from an impl.  Returns the
  // resulting specific interface.
  auto EvaluateIdentifiedWitnesses(Context& context, SemIR::LocId loc_id) const
      -> SemIR::SpecificInterface;
  // Replaces all IdentifiedWitnesses in the interface's specific based on the
  // behaviour of a callback that is run as each witness is visited. Also gives
  // the opportunity to replace ImplSelfWitness in the same substitution pass.
  auto SubstIdentifiedWitnesses(Context& context, SemIR::LocId loc_id,
                                SubstIdentifiedWitnessesCallbacks& callbacks)
      const -> SemIR::SpecificInterface;

  // Returns the interface with a specific that may have IdentifiedWitnesses
  // present. As a result, the specific ID can't be used for comparisons,
  // and does not represent the final evaluated specific.
  auto RawValue() const -> SemIR::SpecificInterface { return interface_; }

  friend auto operator==(const InterfaceWithIdentifiedWitnesses& lhs,
                         const InterfaceWithIdentifiedWitnesses& rhs)
      -> bool = default;

 private:
  SemIR::SpecificInterface interface_;
};

// The IdentifiedFacetType represents all of the interfaces required by a facet
// type against a given Self type, and any other types it constrains. The order
// of the interfaces is fixed for a given facet type, and can thus be used as a
// key for storing and finding witnesses or other data associated with a facet
// type.
struct IdentifiedFacetType {
  // A requirement that `self_facet` implements the `interface`.
  struct RequiredImpl {
    ConstantWithIdentifiedWitnesses self_facet;
    InterfaceWithIdentifiedWitnesses interface;

    friend auto operator==(const RequiredImpl& lhs, const RequiredImpl& rhs)
        -> bool = default;
  };
  // A requirement that assigns `rhs` to `lhs`, where `lhs` will always be an
  // ImplWitnessAccess in the raw value.
  struct Rewrite {
    ConstantWithIdentifiedWitnesses lhs;
    ConstantWithIdentifiedWitnesses rhs;

    friend auto operator==(const Rewrite& lhs, const Rewrite& rhs)
        -> bool = default;
  };
  // A requirement that asserts the `lhs` and `rhs` are the same constant value.
  struct Equivalent {
    ConstantWithIdentifiedWitnesses lhs;
    ConstantWithIdentifiedWitnesses rhs;

    friend auto operator==(const Equivalent& lhs, const Equivalent& rhs)
        -> bool = default;
  };

  IdentifiedFacetType(IdentifiedFacetTypeKey key, bool partially_identified,
                      llvm::ArrayRef<RequiredImpl> extends,
                      llvm::ArrayRef<RequiredImpl> self_impls,
                      llvm::ArrayRef<Rewrite> rewrites,
                      llvm::ArrayRef<Equivalent> equivalents);

  // The order here defines the order of impl witnesses for this facet type.
  auto required_impls() const -> llvm::ArrayRef<RequiredImpl> {
    return required_impls_;
  }

  auto rewrites() const -> llvm::ArrayRef<Rewrite> { return rewrites_; }
  auto equivalents() const -> llvm::ArrayRef<Equivalent> {
    return equivalents_;
  }

  auto partially_identified() const -> bool {
    return key_.num_require_impls >= 0;
  }

  class ImplAsTarget {
   public:
    explicit ImplAsTarget(llvm::SmallVector<SemIR::SpecificInterface> targets)
        : targets_(std::move(targets)) {}

    // Whether this IdentifiedFacetType is a valid `impl as` target, which means
    // that it has one extend impls requirement.
    auto is_valid() const -> bool { return targets_.size() == 1; }
    // The `impl as` target interface, if `is_valid()` is true.
    auto target() const -> SemIR::SpecificInterface {
      CARBON_CHECK(is_valid());
      return targets_.front();
    }
    // The number of extend impls requirements found, if `is_valid()` is false.
    auto num_interfaces() const -> int { return targets_.size(); }

   private:
    llvm::SmallVector<SemIR::SpecificInterface> targets_;
  };

  // Replaces all IdentifiedWitnesses with LookupImplWitnesses in extend
  // requirements, which will evaluate and perform an impl lookup. Then
  // deduplicates the resulting interfaces and returns the single `impl as`
  // target interface if possible.
  auto EvaluateImplWitnessesToGetImplAsTarget(Context& context,
                                              SemIR::LocId loc_id) const
      -> ImplAsTarget;

  auto GetAsKey() const -> IdentifiedFacetTypeKey { return key_; }

 private:
  IdentifiedFacetTypeKey key_;

  // Requirements that a facet value implements an interface, mentioned
  // explicitly in the facet type expression or transitively through a named
  // constraint.
  llvm::SmallVector<RequiredImpl> required_impls_;

  // The subset of interfaces in `required_impls_` that extend the facet type.
  // These need to be deduplicated after replacing IdentifiedWitnesses in them
  // to determine the single extended interface, if any.
  llvm::SmallVector<InterfaceWithIdentifiedWitnesses> extend_impls_;

  // Rewrite requirements from a facet type and any named constraints that it
  // references, flattened to a single list. Only rewrite requirements from
  // extended require decls are included here. Rewrite requirements from a
  // non-extended require decl are treated as same-type requirements.
  llvm::SmallVector<Rewrite> rewrites_;

  // Same-type requirements from a facet type and any named constraints that it
  // references, flattened to a single list. This includes rewrite requirements
  // from non-extended require decls, which are treated as being same-type
  // requirements.
  llvm::SmallVector<Equivalent> equivalents_;
};

// Helper for evaluating a part of a requirement from an IdentifiedFacetType.
// For cases where you need to work with the `RawValue()` and selectively
// evaluate a part of it.
auto EvaluateIdentifiedWitnesses(Context& context, SemIR::LocId loc_id,
                                 SemIR::SpecificInterface interface)
    -> SemIR::SpecificInterface;

// Given an array of witnesses, sorts them to match the ordering of the specific
// interfaces in the IdentifiedFacetType that produced the witness set, which is
// the canonical witness order, and returns the resulting block ID. This assumes
// witnesses have already been deduplicated, and do not contain errors, because
// it's mainly for imports.
auto AddCanonicalWitnessesBlock(Context& context,
                                llvm::SmallVector<SemIR::InstId>& witnesses)
    -> SemIR::InstBlockId;

using IdentifiedFacetTypeStore =
    CanonicalValueStore<IdentifiedFacetTypeId, IdentifiedFacetTypeKey,
                        Tag<SemIR::CheckIRId>, IdentifiedFacetType>;

}  // namespace Carbon::Check

namespace Carbon {
extern template class CanonicalValueStore<
    Check::IdentifiedFacetTypeId, Check::IdentifiedFacetTypeKey,
    Tag<SemIR::CheckIRId>, Check::IdentifiedFacetType>;
extern template class ValueStore<Check::IdentifiedFacetTypeId,
                                 Check::IdentifiedFacetType,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_CHECK_IDENTIFIED_FACET_TYPE_H_
