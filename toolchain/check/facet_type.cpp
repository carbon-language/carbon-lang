// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/facet_type.h"

#include <compare>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto FacetTypeFromInterface(Context& context, SemIR::InterfaceId interface_id,
                            SemIR::SpecificId specific_id) -> SemIR::FacetType {
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(
      SemIR::FacetTypeInfo{.extend_constraints = {{interface_id, specific_id}},
                           .other_requirements = false});
  return {.type_id = SemIR::TypeType::TypeId, .facet_type_id = facet_type_id};
}

// Returns whether the `LookupImplWitness` of `witness_id` matches `interface`.
static auto WitnessQueryMatchesInterface(
    Context& context, SemIR::InstId witness_id,
    const SemIR::SpecificInterface& interface) -> bool {
  auto lookup = context.insts().GetAs<SemIR::LookupImplWitness>(witness_id);
  return interface ==
         context.specific_interfaces().Get(lookup.query_specific_interface_id);
}

static auto IncompleteFacetTypeDiagnosticBuilder(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id,
    bool is_definition) -> DiagnosticBuilder {
  if (is_definition) {
    CARBON_DIAGNOSTIC(ImplAsIncompleteFacetTypeDefinition, Error,
                      "definition of impl as incomplete facet type {0}",
                      InstIdAsType);
    return context.emitter().Build(loc_id, ImplAsIncompleteFacetTypeDefinition,
                                   facet_type_inst_id);
  } else {
    CARBON_DIAGNOSTIC(
        ImplAsIncompleteFacetTypeRewrites, Error,
        "declaration of impl as incomplete facet type {0} with rewrites",
        InstIdAsType);
    return context.emitter().Build(loc_id, ImplAsIncompleteFacetTypeRewrites,
                                   facet_type_inst_id);
  }
}

auto InitialFacetTypeImplWitness(
    Context& context, SemIR::LocId witness_loc_id,
    SemIR::TypeInstId facet_type_inst_id, SemIR::TypeInstId self_type_inst_id,
    const SemIR::SpecificInterface& interface_to_witness,
    SemIR::SpecificId self_specific_id, bool is_definition) -> SemIR::InstId {
  // TODO: Finish facet type resolution. This code currently only handles
  // rewrite constraints that set associated constants to a concrete value.
  // Need logic to topologically sort rewrites to respect dependencies, and
  // afterwards reject duplicates that are not identical.

  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  CARBON_CHECK(facet_type_id != SemIR::ErrorInst::TypeId);
  auto facet_type = context.types().GetAs<SemIR::FacetType>(facet_type_id);
  // TODO: This is currently a copy because I'm not sure whether anything could
  // cause the facet type store to resize before we are done with it.
  auto facet_type_info = context.facet_types().Get(facet_type.facet_type_id);

  if (!is_definition && facet_type_info.rewrite_constraints.empty()) {
    auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
        context, witness_loc_id,
        {.elements_id = context.inst_blocks().AddPlaceholder(),
         .impl_id = SemIR::ImplId::None});
    return AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .witness_table_id = witness_table_inst_id,
         .specific_id = self_specific_id});
  }

  if (!RequireCompleteType(
          context, facet_type_id, SemIR::LocId(facet_type_inst_id), [&] {
            return IncompleteFacetTypeDiagnosticBuilder(
                context, witness_loc_id, facet_type_inst_id, is_definition);
          })) {
    return SemIR::ErrorInst::InstId;
  }

  const auto& interface =
      context.interfaces().Get(interface_to_witness.interface_id);
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  // TODO: When this function is used for things other than just impls, may want
  // to only load the specific associated entities that are mentioned in rewrite
  // rules.
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  SemIR::InstId witness_inst_id = SemIR::InstId::None;
  llvm::MutableArrayRef<SemIR::InstId> table;
  {
    auto elements_id =
        context.inst_blocks().AddUninitialized(assoc_entities.size());
    table = context.inst_blocks().GetMutable(elements_id);
    for (auto& uninit : table) {
      uninit = SemIR::ImplWitnessTablePlaceholder::TypeInstId;
    }

    auto witness_table_inst_id = AddInst<SemIR::ImplWitnessTable>(
        context, witness_loc_id,
        {.elements_id = elements_id, .impl_id = SemIR::ImplId::None});

    witness_inst_id = AddInst<SemIR::ImplWitness>(
        context, witness_loc_id,
        {.type_id = GetSingletonType(context, SemIR::WitnessType::TypeInstId),
         .witness_table_id = witness_table_inst_id,
         .specific_id = self_specific_id});
  }

  for (auto rewrite : facet_type_info.rewrite_constraints) {
    auto access =
        context.insts().GetAs<SemIR::ImplWitnessAccess>(rewrite.lhs_id);
    if (!WitnessQueryMatchesInterface(context, access.witness_id,
                                      interface_to_witness)) {
      continue;
    }
    auto& table_entry = table[access.index.index];
    if (table_entry == SemIR::ErrorInst::InstId) {
      // Don't overwrite an error value. This prioritizes not generating
      // multiple errors for one associated constant over picking a value
      // for it to use to attempt recovery.
      continue;
    }
    auto rewrite_inst_id = rewrite.rhs_id;
    if (rewrite_inst_id == SemIR::ErrorInst::InstId) {
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    auto decl_id = context.constant_values().GetConstantInstId(
        assoc_entities[access.index.index]);
    CARBON_CHECK(decl_id.has_value(), "Non-constant associated entity");
    if (decl_id == SemIR::ErrorInst::InstId) {
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    auto assoc_constant_decl =
        context.insts().TryGetAs<SemIR::AssociatedConstantDecl>(decl_id);
    if (!assoc_constant_decl) {
      auto type_id = context.insts().Get(decl_id).type_id();
      auto type_inst = context.types().GetAsInst(type_id);
      auto fn_type = type_inst.As<SemIR::FunctionType>();
      const auto& fn = context.functions().Get(fn_type.function_id);
      CARBON_DIAGNOSTIC(RewriteForAssociatedFunction, Error,
                        "rewrite specified for associated function {0}",
                        SemIR::NameId);
      context.emitter().Emit(facet_type_inst_id, RewriteForAssociatedFunction,
                             fn.name_id);
      table_entry = SemIR::ErrorInst::InstId;
      continue;
    }

    // FacetTypes resolution disallows two rewrites to the same associated
    // constant, so we won't ever have a facet write twice to the same position
    // in the witness table.
    CARBON_CHECK(table_entry == SemIR::ImplWitnessTablePlaceholder::TypeInstId);

    // If the associated constant has a symbolic type, convert the rewrite
    // value to that type now we know the value of `Self`.
    SemIR::TypeId assoc_const_type_id = assoc_constant_decl->type_id;
    if (assoc_const_type_id.is_symbolic()) {
      // Get the type of the associated constant in this interface with this
      // value for `Self`.
      assoc_const_type_id = GetTypeForSpecificAssociatedEntity(
          context, SemIR::LocId(facet_type_inst_id),
          interface_to_witness.specific_id, decl_id,
          context.types().GetTypeIdForTypeInstId(self_type_inst_id),
          witness_inst_id);
      // Perform the conversion of the value to the type. We skipped this when
      // forming the facet type because the type of the associated constant
      // was symbolic.
      auto converted_inst_id =
          ConvertToValueOfType(context, SemIR::LocId(facet_type_inst_id),
                               rewrite_inst_id, assoc_const_type_id);
      // Canonicalize the converted constant value.
      converted_inst_id =
          context.constant_values().GetConstantInstId(converted_inst_id);
      // The result of conversion can be non-constant even if the original
      // value was constant.
      if (converted_inst_id.has_value()) {
        rewrite_inst_id = converted_inst_id;
      } else {
        const auto& assoc_const = context.associated_constants().Get(
            assoc_constant_decl->assoc_const_id);
        CARBON_DIAGNOSTIC(
            AssociatedConstantNotConstantAfterConversion, Error,
            "associated constant {0} given value {1} that is not constant "
            "after conversion to {2}",
            SemIR::NameId, InstIdAsConstant, SemIR::TypeId);
        context.emitter().Emit(
            facet_type_inst_id, AssociatedConstantNotConstantAfterConversion,
            assoc_const.name_id, rewrite_inst_id, assoc_const_type_id);
        rewrite_inst_id = SemIR::ErrorInst::InstId;
      }
    }

    CARBON_CHECK(rewrite_inst_id == context.constant_values().GetConstantInstId(
                                        rewrite_inst_id),
                 "Rewritten value for associated constant is not canonical.");

    table_entry = AddInst<SemIR::ImplWitnessAssociatedConstant>(
        context, witness_loc_id,
        {.type_id = context.insts().Get(rewrite_inst_id).type_id(),
         .inst_id = rewrite_inst_id});
  }
  return witness_inst_id;
}

auto RequireCompleteFacetTypeForImplDefinition(
    Context& context, SemIR::LocId loc_id, SemIR::TypeInstId facet_type_inst_id)
    -> bool {
  auto facet_type_id =
      context.types().GetTypeIdForTypeInstId(facet_type_inst_id);
  return RequireCompleteType(
      context, facet_type_id, SemIR::LocId(facet_type_inst_id), [&] {
        return IncompleteFacetTypeDiagnosticBuilder(context, loc_id,
                                                    facet_type_inst_id,
                                                    /*is_definition=*/true);
      });
}

auto AllocateFacetTypeImplWitness(Context& context,
                                  SemIR::InterfaceId interface_id,
                                  SemIR::InstBlockId witness_id) -> void {
  const auto& interface = context.interfaces().Get(interface_id);
  CARBON_CHECK(interface.is_complete());
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
  }

  llvm::SmallVector<SemIR::InstId> empty_table(
      assoc_entities.size(), SemIR::ImplWitnessTablePlaceholder::TypeInstId);
  context.inst_blocks().ReplacePlaceholder(witness_id, empty_table);
}

namespace {
struct FacetTypeConstraintValue {
  SemIR::EntityNameId entity_name_id;
  SemIR::ElementIndex access_index;
  SemIR::SpecificInterfaceId specific_interface_id;

  friend auto operator==(const FacetTypeConstraintValue& lhs,
                         const FacetTypeConstraintValue& rhs) -> bool = default;
};
}  // namespace

static auto GetFacetTypeConstraintValue(Context& context,
                                        SemIR::ImplWitnessAccess access)
    -> std::optional<FacetTypeConstraintValue> {
  auto lookup =
      context.insts().TryGetAs<SemIR::LookupImplWitness>(access.witness_id);
  if (lookup) {
    auto self = context.insts().TryGetAs<SemIR::BindSymbolicName>(
        context.constant_values().GetConstantInstId(
            lookup->query_self_inst_id));
    if (self) {
      return {{.entity_name_id = self->entity_name_id,
               .access_index = access.index,
               .specific_interface_id = lookup->query_specific_interface_id}};
    }
  }
  return std::nullopt;
}

// Returns true if two values in a rewrite constraint are equivalent. Two
// `ImplWitnessAccess` instructions that refer to the same associated constant
// through the same facet value are treated as equivalent.
static auto CompareFacetTypeConstraintValues(Context& context,
                                             SemIR::InstId lhs_id,
                                             SemIR::InstId rhs_id) -> bool {
  if (lhs_id == rhs_id) {
    return true;
  }

  auto lhs_access = context.insts().TryGetAs<SemIR::ImplWitnessAccess>(lhs_id);
  auto rhs_access = context.insts().TryGetAs<SemIR::ImplWitnessAccess>(rhs_id);
  if (lhs_access && rhs_access) {
    auto lhs_access_value = GetFacetTypeConstraintValue(context, *lhs_access);
    auto rhs_access_value = GetFacetTypeConstraintValue(context, *rhs_access);
    // We do *not* want to get the evaluated result of `ImplWitnessAccess` here,
    // we want to keep them as a reference to an associated constant for the
    // resolution phase.
    return lhs_access_value && rhs_access_value &&
           *lhs_access_value == *rhs_access_value;
  }

  return context.constant_values().GetConstantInstId(lhs_id) ==
         context.constant_values().GetConstantInstId(rhs_id);
}

// A mapping of each associated constant (represented as `ImplWitnessAccess`) to
// its value (represented as an `InstId`). Used to track rewrite constraints,
// with the LHS mapping to the resolved value of the RHS.
class AccessRewriteValues {
 public:
  enum State {
    NotRewritten,
    BeingRewritten,
    FullyRewritten,
  };
  struct Value {
    State state;
    SemIR::InstId inst_id;
  };

  auto InsertNotRewritten(Context& context, SemIR::ImplWitnessAccess access,
                          SemIR::InstId inst_id) -> void {
    map_.insert({*GetKey(context, access), {NotRewritten, inst_id}});
  }

  // Finds and returns a pointer into the cache for a given ImplWitnessAccess.
  // The pointer will be invalidated by mutating the cache. Returns `nullptr`
  // if `access` is not found.
  auto FindRef(Context& context, SemIR::ImplWitnessAccess access) -> Value* {
    auto key = GetKey(context, access);
    if (!key) {
      return nullptr;
    }
    auto it = map_.find(*key);
    if (it == map_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  auto SetBeingRewritten(Value& value) -> void {
    if (value.state == NotRewritten) {
      value.state = BeingRewritten;
    }
  }

  auto SetFullyRewritten(Context& context, Value& value, SemIR::InstId inst_id)
      -> void {
    CARBON_CHECK(
        value.state == BeingRewritten ||
        CompareFacetTypeConstraintValues(context, value.inst_id, inst_id));
    value = {FullyRewritten, inst_id};
  }

 private:
  using Key = FacetTypeConstraintValue;
  struct KeyInfo {
    static auto getEmptyKey() -> Key {
      return {
          .entity_name_id = SemIR::EntityNameId::None,
          .access_index = SemIR::ElementIndex(-1),
          .specific_interface_id = SemIR::SpecificInterfaceId::None,
      };
    }
    static auto getTombstoneKey() -> Key {
      return {
          .entity_name_id = SemIR::EntityNameId::None,
          .access_index = SemIR::ElementIndex(-2),
          .specific_interface_id = SemIR::SpecificInterfaceId::None,
      };
    }
    static auto getHashValue(Key key) -> unsigned {
      // This hash produces the same value if two ImplWitnessAccess are
      // pointing to the same associated constant, even if they are different
      // instruction ids.
      //
      // TODO: This truncates the high bits of the hash code which does not
      // make for a good hash function.
      return static_cast<unsigned>(static_cast<uint64_t>(HashValue(key)));
    }
    static auto isEqual(Key lhs, Key rhs) -> bool {
      // This comparison is true if the two ImplWitnessAccess are pointing to
      // the same associated constant, even if they are different instruction
      // ids.
      return lhs == rhs;
    }
  };

  // Returns a key for the `access` to an associated context if the access is
  // through a facet value. If the access it through another `ImplWitnessAccess`
  // then no key is able to be made.
  auto GetKey(Context& context, SemIR::ImplWitnessAccess access)
      -> std::optional<Key> {
    return GetFacetTypeConstraintValue(context, access);
  }

  // Try avoid heap allocations in the common case where there are a small
  // number of rewrite rules referring to each other by keeping up to 16 on
  // the stack.
  //
  // TODO: Revisit if 16 is an appropriate number when we can measure how deep
  // rewrite constraint chains go in practice.
  llvm::SmallDenseMap<Key, Value, 16, KeyInfo> map_;
};

// To be used for substituting into the RHS of a rewrite constraint.
//
// It will substitute any `ImplWitnessAccess` into `.Self` (a reference to an
// associated constant) with the RHS of another rewrite constraint that writes
// to the same associated constant. For example:
// ```
// Z where .X = () and .Y = .X
// ```
// Here the second `.X` is an `ImplWitnessAccess` which would be substituted by
// finding the first rewrite constraint, where the LHS is for the same
// associated constant and using its RHS. So the substitution would produce:
// ```
// Z where .X = () and .Y = ()
// ```
//
// This additionally diagnoses cycles when the `ImplWitnessAccess` is reading
// from the same rewrite constraint, and is thus assigning to the associated
// constant a value that refers to the same associated constant, such as with `Z
// where .X = C(.X)`. In the event of a cycle, the `ImplWitnessAccess` is
// replaced with `ErrorInst` so that further evaluation of the
// `ImplWitnessAccess` will not loop infinitely.
//
// The `rewrite_values` given to the constructor must be set up initially with
// each rewrite rule of an associated constant inserted with its unresolved
// value via `InsertNotRewritten`. Then for each rewrite rule of an associated
// constant, the LHS access should be set as being rewritten with its state
// changed to `BeingRewritten` in order to detect cycles before performing
// SubstInst. The result of SubstInst should be preserved afterward by changing
// the state and value for the LHS to `FullyRewritten` and the subst output
// instruction, respectively, to avoid duplicating work.
class SubstImplWitnessAccessCallbacks : public SubstInstCallbacks {
 public:
  explicit SubstImplWitnessAccessCallbacks(Context* context,
                                           SemIR::LocId loc_id,
                                           AccessRewriteValues* rewrite_values)
      : SubstInstCallbacks(context),
        loc_id_(loc_id),
        rewrite_values_(rewrite_values) {}

  auto Subst(SemIR::InstId& rhs_inst_id) -> SubstResult override {
    auto rhs_access =
        context().insts().TryGetAs<SemIR::ImplWitnessAccess>(rhs_inst_id);
    if (!rhs_access) {
      // We only want to substitute ImplWitnessAccesses written directly on the
      // RHS of the rewrite constraint, not when they are nested inside facet
      // types that are part of the RHS, like `.X = C as (I where .Y = {})`.
      if (context().insts().Is<SemIR::FacetType>(rhs_inst_id)) {
        return SubstResult::FullySubstituted;
      }
      if (context().constant_values().Get(rhs_inst_id).is_concrete()) {
        // There's no ImplWitnessAccess that we care about inside this
        // instruction.
        return SubstResult::FullySubstituted;
      } else {
        // SubstOperands will result in a Rebuild or ReuseUnchanged callback, so
        // push the non-ImplWitnessAccess to get proper bracketing, allowing us
        // to pop it in the paired callback.
        substs_in_progress_.push_back(rhs_inst_id);
        return SubstResult::SubstOperands;
      }
    }

    // If the access is going through a nested `ImplWitnessAccess`, that
    // access needs to be resolved to a facet value first. If it can't be
    // resolved then the outer one can not be either.
    if (auto lookup = context().insts().TryGetAs<SemIR::LookupImplWitness>(
            rhs_access->witness_id)) {
      if (context().insts().Is<SemIR::ImplWitnessAccess>(
              lookup->query_self_inst_id)) {
        substs_in_progress_.push_back(rhs_inst_id);
        return SubstResult::SubstOperandsAndRetry;
      }
    }

    auto* rewrite_value = rewrite_values_->FindRef(context(), *rhs_access);
    if (!rewrite_value) {
      // The RHS refers to an associated constant for which there is no rewrite
      // rule.
      return SubstResult::FullySubstituted;
    }

    // Diagnose a cycle if the RHS refers to something that depends on the value
    // of the RHS.
    if (rewrite_value->state == AccessRewriteValues::BeingRewritten) {
      CARBON_DIAGNOSTIC(FacetTypeConstraintCycle, Error,
                        "found cycle in facet type constraint for {0}",
                        InstIdAsConstant);
      // TODO: It would be nice to note the places where the values are
      // assigned but rewrite constraint instructions are from canonical
      // constant values, and have no locations. We'd need to store a location
      // along with them in the rewrite constraints, and track propagation of
      // locations here, which may imply heap allocations.
      context().emitter().Emit(loc_id_, FacetTypeConstraintCycle, rhs_inst_id);
      rhs_inst_id = SemIR::ErrorInst::InstId;
      return SubstResult::FullySubstituted;
    } else if (rewrite_value->state == AccessRewriteValues::FullyRewritten) {
      rhs_inst_id = rewrite_value->inst_id;
      return SubstResult::FullySubstituted;
    }

    // We have a non-rewritten RHS. We need to recurse on rewriting it. Reuse
    // the previous lookup by mutating it in place.
    rewrite_values_->SetBeingRewritten(*rewrite_value);

    // The ImplWitnessAccess was replaced with some other instruction, which may
    // contain or be another ImplWitnessAccess. Keep track of the associated
    // constant we are now computing the value of.
    substs_in_progress_.push_back(rhs_inst_id);
    rhs_inst_id = rewrite_value->inst_id;
    return SubstResult::SubstAgain;
  }

  auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst new_inst)
      -> SemIR::InstId override {
    auto inst_id = RebuildNewInst(loc_id_, new_inst);
    auto subst_inst_id = substs_in_progress_.pop_back_val();
    if (auto access = context().insts().TryGetAs<SemIR::ImplWitnessAccess>(
            subst_inst_id)) {
      if (auto* rewrite_value = rewrite_values_->FindRef(context(), *access)) {
        rewrite_values_->SetFullyRewritten(context(), *rewrite_value, inst_id);
      }
    }
    return inst_id;
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) -> SemIR::InstId override {
    auto subst_inst_id = substs_in_progress_.pop_back_val();
    if (auto access = context().insts().TryGetAs<SemIR::ImplWitnessAccess>(
            subst_inst_id)) {
      if (auto* rewrite_value = rewrite_values_->FindRef(context(), *access)) {
        rewrite_values_->SetFullyRewritten(context(), *rewrite_value,
                                           orig_inst_id);
      }
    }
    return orig_inst_id;
  }

 private:
  struct SubstInProgress {
    // The associated constant whose value is being determined, represented as
    // an ImplWitnessAccess. Or another instruction that we are recursing
    // through.
    SemIR::InstId inst_id;
  };

  // The location of the rewrite constraints as a whole.
  SemIR::LocId loc_id_;
  // Tracks the resolved value of each rewrite constraint, keyed by the
  // `ImplWitnessAccess` of the associated constant on the LHS of the
  // constraint. The value of each associated constant may be changed during
  // substitution, replaced with a fully resolved value for the RHS. This allows
  // us to cache work; when a value for an associated constant is found once it
  // can be reused cheaply, avoiding exponential runtime when rewrite rules
  // refer to each other in ways that create exponential references.
  AccessRewriteValues* rewrite_values_;
  // A stack of instructions being replaced in Subst(). When it's an associated
  // constant, then it represents the constant value is being determined,
  // represented as an ImplWitnessAccess.
  //
  // Avoid heap allocations in common cases, if there are chains of instructions
  // in associated constants with a depth at most 16.
  llvm::SmallVector<SemIR::InstId, 16> substs_in_progress_;
};

auto ResolveFacetTypeRewriteConstraints(
    Context& context, SemIR::LocId loc_id,
    llvm::SmallVector<SemIR::FacetTypeInfo::RewriteConstraint>& rewrites)
    -> bool {
  if (rewrites.empty()) {
    return true;
  }

  AccessRewriteValues rewrite_values;

  for (auto& constraint : rewrites) {
    auto lhs_access =
        context.insts().TryGetAs<SemIR::ImplWitnessAccess>(constraint.lhs_id);
    if (!lhs_access) {
      continue;
    }

    rewrite_values.InsertNotRewritten(context, *lhs_access, constraint.rhs_id);
  }

  for (auto& constraint : rewrites) {
    auto lhs_access =
        context.insts().TryGetAs<SemIR::ImplWitnessAccess>(constraint.lhs_id);
    if (!lhs_access) {
      continue;
    }

    auto* lhs_rewrite_value = rewrite_values.FindRef(context, *lhs_access);
    // Every LHS was added with InsertNotRewritten above.
    CARBON_CHECK(lhs_rewrite_value);
    rewrite_values.SetBeingRewritten(*lhs_rewrite_value);

    auto replace_witness_callbacks =
        SubstImplWitnessAccessCallbacks(&context, loc_id, &rewrite_values);
    auto rhs_subst_inst_id =
        SubstInst(context, constraint.rhs_id, replace_witness_callbacks);
    if (rhs_subst_inst_id == SemIR::ErrorInst::InstId) {
      return false;
    }

    if (lhs_rewrite_value->state == AccessRewriteValues::FullyRewritten &&
        !CompareFacetTypeConstraintValues(context, lhs_rewrite_value->inst_id,
                                          rhs_subst_inst_id)) {
      if (lhs_rewrite_value->inst_id != SemIR::ErrorInst::InstId) {
        CARBON_DIAGNOSTIC(AssociatedConstantWithDifferentValues, Error,
                          "associated constant {0} given two different "
                          "values {1} and {2}",
                          InstIdAsConstant, InstIdAsConstant, InstIdAsConstant);
        // Use inst id ordering as a simple proxy for source ordering, to
        // try to name the values in the same order they appear in the facet
        // type.
        auto source_order1 =
            lhs_rewrite_value->inst_id.index < rhs_subst_inst_id.index
                ? lhs_rewrite_value->inst_id
                : rhs_subst_inst_id;
        auto source_order2 =
            lhs_rewrite_value->inst_id.index >= rhs_subst_inst_id.index
                ? lhs_rewrite_value->inst_id
                : rhs_subst_inst_id;
        // TODO: It would be nice to note the places where the values are
        // assigned but rewrite constraint instructions are from canonical
        // constant values, and have no locations. We'd need to store a
        // location along with them in the rewrite constraints.
        context.emitter().Emit(loc_id, AssociatedConstantWithDifferentValues,
                               constraint.lhs_id, source_order1, source_order2);
      }
      return false;
    }

    rewrite_values.SetFullyRewritten(context, *lhs_rewrite_value,
                                     rhs_subst_inst_id);
  }

  // Rebuild the `rewrites` vector with resolved values for the RHS. Drop any
  // duplicate rewrites in the `rewrites` vector by walking through the
  // `rewrite_values` map and dropping the computed RHS value for each LHS the
  // first time we see it, and erasing the constraint from the vector if we see
  // the same LHS again.
  size_t keep_size = rewrites.size();
  for (size_t i = 0; i < keep_size;) {
    auto& constraint = rewrites[i];

    auto lhs_access =
        context.insts().TryGetAs<SemIR::ImplWitnessAccess>(constraint.lhs_id);
    if (!lhs_access) {
      ++i;
      continue;
    }

    auto& rewrite_value = *rewrite_values.FindRef(context, *lhs_access);
    auto rhs_id = std::exchange(rewrite_value.inst_id, SemIR::InstId::None);
    if (rhs_id == SemIR::InstId::None) {
      std::swap(rewrites[i], rewrites[keep_size - 1]);
      --keep_size;
    } else {
      rewrites[i].rhs_id = rhs_id;
      ++i;
    }
  }
  rewrites.erase(rewrites.begin() + keep_size, rewrites.end());

  return true;
}

}  // namespace Carbon::Check
