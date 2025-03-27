// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic.h"

#include "common/map.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic_region_stack.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

static auto MakeSelfSpecificId(Context& context, SemIR::GenericId generic_id)
    -> SemIR::SpecificId;

auto StartGenericDecl(Context& context) -> void {
  context.generic_region_stack().Push();
}

auto StartGenericDefinition(Context& context) -> void {
  // Push a generic region even if we don't have a generic_id. We might still
  // have locally-introduced generic parameters to track:
  //
  // fn F() {
  //   let T:! type = i32;
  //   var x: T;
  // }
  context.generic_region_stack().Push();
}

// Adds an instruction `generic_inst_id` to the eval block for a generic region,
// which is the current instruction block. The instruction `generic_inst_id` is
// expected to compute the value of the constant described by `const_inst_id` in
// each specific. Forms and returns a corresponding symbolic constant ID that
// refers to the substituted value of that instruction in each specific.
static auto AddGenericConstantInstToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region, SemIR::InstId const_inst_id,
    SemIR::InstId generic_inst_id, SemIR::ConstantDependence dependence)
    -> SemIR::ConstantId {
  auto index = SemIR::GenericInstIndex(
      region, context.inst_block_stack().PeekCurrentBlockContents().size());
  context.inst_block_stack().AddInstId(generic_inst_id);
  return context.constant_values().AddSymbolicConstant(
      {.inst_id = const_inst_id,
       .generic_id = generic_id,
       .index = index,
       .dependence = dependence});
}

namespace {
// A map from an instruction ID representing a canonical symbolic constant to an
// instruction within an eval block of the generic that computes the specific
// value for that constant.
//
// We arbitrarily use a small size of 256 bytes for the map.
// TODO: Determine a better number based on measurements.
using ConstantsInGenericMap = Map<SemIR::InstId, SemIR::InstId, 256>;

// Substitution callbacks to rebuild a generic constant in the eval block for a
// generic region.
class RebuildGenericConstantInEvalBlockCallbacks : public SubstInstCallbacks {
 public:
  RebuildGenericConstantInEvalBlockCallbacks(
      Context& context, SemIR::GenericId generic_id,
      SemIR::GenericInstIndex::Region region, SemIR::LocId loc_id,
      ConstantsInGenericMap& constants_in_generic, bool inside_redeclaration)
      : context_(context),
        generic_id_(generic_id),
        region_(region),
        loc_id_(loc_id),
        constants_in_generic_(constants_in_generic),
        inside_redeclaration_(inside_redeclaration) {}

  auto context() const -> Context& { return context_; }

  // Check for instructions for which we already have a mapping into the eval
  // block, and substitute them for the instructions in the eval block.
  auto Subst(SemIR::InstId& inst_id) const -> bool override {
    auto const_id = context_.constant_values().Get(inst_id);
    if (!const_id.has_value()) {
      // An unloaded import ref should never contain anything we need to
      // substitute into. Don't trigger loading it here.
      CARBON_CHECK(
          context_.insts().Is<SemIR::ImportRefUnloaded>(inst_id),
          "Substituting into instruction with invalid constant ID: {0}",
          context_.insts().Get(inst_id));
      return true;
    }
    if (!context_.constant_values().DependsOnGenericParameter(const_id)) {
      // This instruction doesn't have a symbolic constant value, so can't
      // contain any bindings that need to be substituted.
      return true;
    }

    // If this instruction is in the map, return the known result.
    if (auto result = constants_in_generic_.Lookup(
            context_.constant_values().GetInstId(const_id))) {
      // In order to reuse instructions from the generic as often as possible,
      // keep this instruction as-is if it already has the desired symbolic
      // constant value.
      if (const_id != context_.constant_values().Get(result.value())) {
        inst_id = result.value();
      }
      CARBON_CHECK(inst_id.has_value());
      return true;
    }

    return false;
  }

  // Build a new instruction in the eval block corresponding to the given
  // constant.
  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst) const
      -> SemIR::InstId override {
    auto& orig_symbolic_const = context_.constant_values().GetSymbolicConstant(
        context_.constant_values().Get(orig_inst_id));
    auto const_inst_id = orig_symbolic_const.inst_id;
    auto dependence = orig_symbolic_const.dependence;

    // We might already have an instruction in the eval block if a transitive
    // operand of this instruction has the same constant value.
    auto result = constants_in_generic_.Insert(const_inst_id, [&] {
      if (inside_redeclaration_) {
        // Adding instructions to a redeclaration causes crashes later since it
        // causes us to produce invalid indices into the original declaration's
        // set of instructions. So we terminate now and avoid adding a new
        // instruction and new index. It should not be possible to create this
        // situation where a generic redeclaration introduces new instructions
        // to the eval block.
        CARBON_FATAL("generic redeclaration differs from previous declaration");
      }

      // TODO: Add a function on `Context` to add the instruction without
      // inserting it into the dependent instructions list or computing a
      // constant value for it.
      // TODO: Is the location we pick here always appropriate for the new
      // instruction?
      auto inst_id = context_.sem_ir().insts().AddInNoBlock(
          SemIR::LocIdAndInst::UncheckedLoc(loc_id_, new_inst));
      auto const_id = AddGenericConstantInstToEvalBlock(
          context_, generic_id_, region_, const_inst_id, inst_id, dependence);
      context_.constant_values().Set(inst_id, const_id);
      return inst_id;
    });
    return result.value();
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) const
      -> SemIR::InstId override {
    auto inst = context_.insts().Get(orig_inst_id);
    CARBON_CHECK(
        inst.Is<SemIR::BindSymbolicName>() ||
            inst.Is<SemIR::SymbolicBindingPattern>(),
        "Instruction {0} has symbolic constant value but no symbolic operands",
        inst);

    // Rebuild the instruction anyway so that it's included in the eval block.
    // TODO: Can we just reuse the instruction in this case?
    return Rebuild(orig_inst_id, inst);
  }

 private:
  Context& context_;
  SemIR::GenericId generic_id_;
  SemIR::GenericInstIndex::Region region_;
  SemIR::LocId loc_id_;
  ConstantsInGenericMap& constants_in_generic_;
  bool inside_redeclaration_;
};

// Substitution callbacks to rebuild a template action. This rebuilds the action
// instruction in-place if it needs to be modified.
class RebuildTemplateActionInEvalBlockCallbacks final
    : public RebuildGenericConstantInEvalBlockCallbacks {
 public:
  RebuildTemplateActionInEvalBlockCallbacks(
      Context& context, SemIR::GenericId generic_id,
      SemIR::GenericInstIndex::Region region, SemIR::LocId loc_id,
      ConstantsInGenericMap& constants_in_generic, bool inside_redeclaration,
      SemIR::InstId action_inst_id)
      : RebuildGenericConstantInEvalBlockCallbacks(context, generic_id, region,
                                                   loc_id, constants_in_generic,
                                                   inside_redeclaration),
        action_inst_id_(action_inst_id) {}

  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst) const
      -> SemIR::InstId override {
    if (orig_inst_id == action_inst_id_) {
      // TODO: We want to ReplaceInstPreservingConstantValue here, but don't
      // want to evaluate the action to check the value hasn't changed.
      context().sem_ir().insts().Set(orig_inst_id, new_inst);
      return orig_inst_id;
    }
    return RebuildGenericConstantInEvalBlockCallbacks::Rebuild(orig_inst_id,
                                                               new_inst);
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) const
      -> SemIR::InstId override {
    if (orig_inst_id == action_inst_id_) {
      return orig_inst_id;
    }
    return RebuildGenericConstantInEvalBlockCallbacks::ReuseUnchanged(
        orig_inst_id);
  }

 private:
  SemIR::InstId action_inst_id_;
};
}  // namespace

// Adds instructions to compute the substituted version of `type_id` in each
// specific into the eval block for the generic, which is the current
// instruction block. Returns a symbolic type ID that refers to the substituted
// type in each specific.
static auto AddGenericTypeToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region, SemIR::LocId loc_id,
    ConstantsInGenericMap& constants_in_generic, bool inside_redeclaration,
    SemIR::TypeId type_id) -> SemIR::TypeId {
  // Substitute into the type's constant instruction and rebuild it in the eval
  // block.
  auto type_inst_id =
      SubstInst(context, context.types().GetInstId(type_id),
                RebuildGenericConstantInEvalBlockCallbacks(
                    context, generic_id, region, loc_id, constants_in_generic,
                    inside_redeclaration));
  return context.types().GetTypeIdForTypeInstId(type_inst_id);
}

// Adds instructions to compute the substituted value of `inst_id` in each
// specific into the eval block for the generic, which is the current
// instruction block. Returns a symbolic constant instruction ID that refers to
// the substituted constant value in each specific.
static auto AddGenericConstantToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region,
    ConstantsInGenericMap& constants_in_generic, bool inside_redeclaration,
    SemIR::InstId inst_id) -> SemIR::ConstantId {
  // Substitute into the constant value and rebuild it in the eval block if
  // we've not encountered it before.
  auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);
  auto callbacks = RebuildGenericConstantInEvalBlockCallbacks(
      context, generic_id, region, context.insts().GetLocId(inst_id),
      constants_in_generic, inside_redeclaration);
  auto new_inst_id = SubstInst(context, const_inst_id, callbacks);
  CARBON_CHECK(new_inst_id != const_inst_id,
               "No substitutions performed for generic constant {0}",
               context.insts().Get(inst_id));
  return context.constant_values().Get(new_inst_id);
}

// Adds an instruction that performs a template action to the eval block for the
// generic. The instruction should not yet have been added to any block. The
// instruction might refer to types and constants that need to be rewritten, so
// substitute into it first.
static auto AddTemplateActionToEvalBlock(
    Context& context, SemIR::GenericId generic_id,
    SemIR::GenericInstIndex::Region region,
    ConstantsInGenericMap& constants_in_generic, bool inside_redeclaration,
    SemIR::InstId inst_id) -> void {
  // Substitute into the constant value and rebuild it in the eval block.
  auto new_inst_id = SubstInst(
      context, inst_id,
      RebuildTemplateActionInEvalBlockCallbacks(
          context, generic_id, region, context.insts().GetLocId(inst_id),
          constants_in_generic, inside_redeclaration, inst_id));
  CARBON_CHECK(new_inst_id == inst_id,
               "Substitution changed InstId of template action");
  constants_in_generic.Insert(inst_id, inst_id);

  // Add the action to the eval block and point its constant value back to its
  // index within the block.
  auto& symbolic_constant = context.constant_values().GetSymbolicConstant(
      context.constant_values().Get(inst_id));
  symbolic_constant.generic_id = generic_id;
  symbolic_constant.index = SemIR::GenericInstIndex(
      region, context.inst_block_stack().PeekCurrentBlockContents().size());
  context.inst_block_stack().AddInstId(inst_id);
}

// Populates a map of constants in a generic from the constants in the
// declaration region, in preparation for building the definition region.
static auto PopulateConstantsFromDeclaration(
    Context& context, SemIR::GenericId generic_id,
    ConstantsInGenericMap& constants_in_generic) {
  // For the definition region, populate constants from the declaration.
  auto decl_eval_block = context.inst_blocks().Get(
      context.generics().Get(generic_id).decl_block_id);
  constants_in_generic.GrowForInsertCount(decl_eval_block.size());
  for (auto inst_id : decl_eval_block) {
    auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);
    auto result = constants_in_generic.Insert(const_inst_id, inst_id);
    CARBON_CHECK(result.is_inserted(),
                 "Duplicate constant in generic decl eval block: {0}",
                 context.insts().Get(const_inst_id));
  }
}

// Builds and returns a block of instructions whose constant values need to be
// evaluated in order to resolve a generic to a specific.
static auto MakeGenericEvalBlock(Context& context, SemIR::GenericId generic_id,
                                 SemIR::GenericInstIndex::Region region,
                                 bool inside_redeclaration)
    -> SemIR::InstBlockId {
  context.inst_block_stack().Push();

  ConstantsInGenericMap constants_in_generic;

  if (region == SemIR::GenericInstIndex::Region::Definition ||
      inside_redeclaration) {
    PopulateConstantsFromDeclaration(context, generic_id, constants_in_generic);
  }

  // The work done in this loop might invalidate iterators into the generic
  // region stack, but shouldn't add new dependent instructions to the current
  // region.
  auto num_dependent_insts =
      context.generic_region_stack().PeekDependentInsts().size();
  for (auto i : llvm::seq(num_dependent_insts)) {
    auto [inst_id, dep_kind] =
        context.generic_region_stack().PeekDependentInsts()[i];

    // If the type is symbolic, replace it with a type specific to this generic.
    if ((dep_kind & GenericRegionStack::DependencyKind::SymbolicType) !=
        GenericRegionStack::DependencyKind::None) {
      auto inst = context.insts().Get(inst_id);
      auto type_id = AddGenericTypeToEvalBlock(
          context, generic_id, region, context.insts().GetLocId(inst_id),
          constants_in_generic, inside_redeclaration, inst.type_id());
      // If the generic declaration is invalid, it can result in an error.
      if (type_id == SemIR::ErrorInst::SingletonTypeId) {
        break;
      }
      // TODO: Eventually, completeness requirements should be modeled as
      // constraints on the generic rather than properties of the type. For now,
      // require the transformed type to be complete if the original was.
      if (context.types().IsComplete(inst.type_id())) {
        CompleteTypeOrCheckFail(context, type_id);
      }
      inst.SetType(type_id);
      context.sem_ir().insts().Set(inst_id, inst);
    }

    // If the instruction has a symbolic constant value, then make a note that
    // we'll need to evaluate this instruction when forming the specific. Update
    // the constant value of the instruction to refer to the result of that
    // eventual evaluation.
    if ((dep_kind & GenericRegionStack::DependencyKind::SymbolicConstant) !=
        GenericRegionStack::DependencyKind::None) {
      // Update the constant value to refer to this generic.
      context.constant_values().Set(
          inst_id, AddGenericConstantToEvalBlock(
                       context, generic_id, region, constants_in_generic,
                       inside_redeclaration, inst_id));
    }

    // If the instruction is a template action, add it directly to this position
    // in the eval block.
    if ((dep_kind & GenericRegionStack::DependencyKind::Template) !=
        GenericRegionStack::DependencyKind::None) {
      AddTemplateActionToEvalBlock(context, generic_id, region,
                                   constants_in_generic, inside_redeclaration,
                                   inst_id);
    }
  }

  CARBON_CHECK(
      num_dependent_insts ==
          context.generic_region_stack().PeekDependentInsts().size(),
      "Building eval block added new dependent insts, for example {0}",
      context.insts().Get(context.generic_region_stack()
                              .PeekDependentInsts()[num_dependent_insts]
                              .inst_id));

  return context.inst_block_stack().Pop();
}

// Builds and returns an eval block, given the list of canonical symbolic
// constants that the instructions in the eval block should produce. This is
// used when importing a generic.
auto RebuildGenericEvalBlock(Context& context, SemIR::GenericId generic_id,
                             SemIR::GenericInstIndex::Region region,
                             llvm::ArrayRef<SemIR::InstId> const_ids)
    -> SemIR::InstBlockId {
  context.inst_block_stack().Push();

  // We say we are not inside a redeclaration since this function is used for
  // import and there's no redeclaration there.
  bool inside_redeclaration = false;

  ConstantsInGenericMap constants_in_generic;

  // For the definition region, populate constants from the declaration.
  if (inside_redeclaration ||
      region == SemIR::GenericInstIndex::Region::Definition) {
    PopulateConstantsFromDeclaration(context, generic_id, constants_in_generic);
  }

  constants_in_generic.GrowForInsertCount(const_ids.size());
  for (auto [i, inst_id] : llvm::enumerate(const_ids)) {
    // Build a constant in the inst block.
    AddGenericConstantToEvalBlock(context, generic_id, region,
                                  constants_in_generic, inside_redeclaration,
                                  inst_id);
    CARBON_CHECK(
        context.inst_block_stack().PeekCurrentBlockContents().size() == i + 1,
        "Produced {0} instructions when importing {1}",
        (context.inst_block_stack().PeekCurrentBlockContents().size() - i),
        context.insts().Get(inst_id));
  }

  return context.inst_block_stack().Pop();
}

auto DiscardGenericDecl(Context& context) -> void {
  context.generic_region_stack().Pop();
}

auto BuildGeneric(Context& context, SemIR::InstId decl_id) -> SemIR::GenericId {
  auto all_bindings =
      context.scope_stack().compile_time_bindings_stack().PeekAllValues();

  if (all_bindings.empty()) {
    CARBON_CHECK(context.generic_region_stack().PeekDependentInsts().empty(),
                 "Have dependent instruction {0} in declaration {1} but no "
                 "compile time bindings are in scope.",
                 context.insts().Get(context.generic_region_stack()
                                         .PeekDependentInsts()
                                         .front()
                                         .inst_id),
                 context.insts().Get(decl_id));
    context.generic_region_stack().Pop();
    return SemIR::GenericId::None;
  }

  // Build the new Generic object. Note that we intentionally do not hold a
  // persistent reference to it throughout this function, because the `generics`
  // collection can have items added to it by import resolution while we are
  // building this generic.
  auto bindings_id = context.inst_blocks().Add(all_bindings);

  SemIR::GenericId generic_id = context.generics().Add(
      SemIR::Generic{.decl_id = decl_id,
                     .bindings_id = bindings_id,
                     .self_specific_id = SemIR::SpecificId::None});
  // MakeSelfSpecificId could cause something to be imported, which would
  // invalidate the return value of `context.generics().Get(generic_id)`.
  auto self_specific_id = MakeSelfSpecificId(context, generic_id);
  context.generics().Get(generic_id).self_specific_id = self_specific_id;
  return generic_id;
}

auto FinishGenericDecl(Context& context, SemIRLoc loc,
                       SemIR::GenericId generic_id) -> void {
  if (!generic_id.has_value()) {
    return;
  }
  auto decl_block_id = MakeGenericEvalBlock(
      context, generic_id, SemIR::GenericInstIndex::Region::Declaration,
      /*inside_redeclaration=*/false);
  context.generic_region_stack().Pop();
  context.generics().Get(generic_id).decl_block_id = decl_block_id;

  ResolveSpecificDeclaration(context, loc,
                             context.generics().GetSelfSpecific(generic_id));
}

auto BuildGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId {
  SemIR::GenericId generic_id = BuildGeneric(context, decl_id);
  if (generic_id.has_value()) {
    FinishGenericDecl(context, decl_id, generic_id);
  }
  return generic_id;
}

auto FinishGenericRedecl(Context& context, SemIR::GenericId generic_id)
    -> void {
  if (!generic_id.has_value()) {
    context.generic_region_stack().Pop();
    return;
  }

  auto definition_block_id = MakeGenericEvalBlock(
      context, generic_id, SemIR::GenericInstIndex::Region::Declaration,
      /*inside_redeclaration=*/true);
  CARBON_CHECK(definition_block_id == SemIR::InstBlockId::Empty);

  context.generic_region_stack().Pop();
}

auto FinishGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void {
  if (!generic_id.has_value()) {
    // TODO: We can have symbolic constants in a context that had a non-generic
    // declaration, for example if there's a local generic let binding in a
    // function definition. Handle this case somehow -- perhaps by forming
    // substituted constant values now.
    context.generic_region_stack().Pop();
    return;
  }

  auto definition_block_id = MakeGenericEvalBlock(
      context, generic_id, SemIR::GenericInstIndex::Region::Definition,
      /*inside_redeclaration=*/false);
  context.generics().Get(generic_id).definition_block_id = definition_block_id;

  context.generic_region_stack().Pop();
}

auto ResolveSpecificDeclaration(Context& context, SemIRLoc loc,
                                SemIR::SpecificId specific_id) -> void {
  // If this is the first time we've formed this specific, evaluate its decl
  // block to form information about the specific.
  if (!context.specifics().Get(specific_id).decl_block_id.has_value()) {
    // Set a placeholder value as the decl block ID so we won't attempt to
    // recursively resolve the same specific.
    context.specifics().Get(specific_id).decl_block_id =
        SemIR::InstBlockId::Empty;

    auto decl_block_id =
        TryEvalBlockForSpecific(context, loc, specific_id,
                                SemIR::GenericInstIndex::Region::Declaration);
    // Note that TryEvalBlockForSpecific may reallocate the list of specifics,
    // so re-lookup the specific here.
    context.specifics().Get(specific_id).decl_block_id = decl_block_id;
  }
}

auto MakeSpecific(Context& context, SemIRLoc loc, SemIR::GenericId generic_id,
                  SemIR::InstBlockId args_id) -> SemIR::SpecificId {
  auto specific_id = context.specifics().GetOrAdd(generic_id, args_id);
  ResolveSpecificDeclaration(context, loc, specific_id);
  return specific_id;
}

auto MakeSpecific(Context& context, SemIRLoc loc, SemIR::GenericId generic_id,
                  llvm::ArrayRef<SemIR::InstId> args) -> SemIR::SpecificId {
  auto args_id = context.inst_blocks().AddCanonical(args);
  return MakeSpecific(context, loc, generic_id, args_id);
}

static auto MakeSelfSpecificId(Context& context, SemIR::GenericId generic_id)
    -> SemIR::SpecificId {
  if (!generic_id.has_value()) {
    return SemIR::SpecificId::None;
  }

  auto& generic = context.generics().Get(generic_id);
  auto args = context.inst_blocks().Get(generic.bindings_id);

  // Form a canonical argument list for the generic.
  llvm::SmallVector<SemIR::InstId> arg_ids;
  arg_ids.reserve(args.size());
  for (auto arg_id : args) {
    arg_ids.push_back(context.constant_values().GetConstantInstId(arg_id));
  }
  auto args_id = context.inst_blocks().AddCanonical(arg_ids);
  return context.specifics().GetOrAdd(generic_id, args_id);
}

auto MakeSelfSpecific(Context& context, SemIRLoc loc,
                      SemIR::GenericId generic_id) -> SemIR::SpecificId {
  // Build a corresponding specific.
  SemIR::SpecificId specific_id = MakeSelfSpecificId(context, generic_id);
  // TODO: This could be made more efficient. We don't need to perform
  // substitution here; we know we want identity mappings for all constants and
  // types. We could also consider not storing the mapping at all in this case.
  ResolveSpecificDeclaration(context, loc, specific_id);
  return specific_id;
}

auto ResolveSpecificDefinition(Context& context, SemIRLoc loc,
                               SemIR::SpecificId specific_id) -> bool {
  // TODO: Handle recursive resolution of the same generic definition.
  auto& specific = context.specifics().Get(specific_id);
  auto generic_id = specific.generic_id;
  CARBON_CHECK(generic_id.has_value(), "Specific with no generic ID");

  if (!specific.definition_block_id.has_value()) {
    // Evaluate the eval block for the definition of the generic.
    auto& generic = context.generics().Get(generic_id);
    if (!generic.definition_block_id.has_value()) {
      // The generic is not defined yet.
      return false;
    }
    auto definition_block_id = TryEvalBlockForSpecific(
        context, loc, specific_id, SemIR::GenericInstIndex::Region::Definition);
    // Note that TryEvalBlockForSpecific may reallocate the list of specifics,
    // so re-lookup the specific here.
    context.specifics().Get(specific_id).definition_block_id =
        definition_block_id;
  }
  return true;
}

auto DiagnoseIfGenericMissingExplicitParameters(
    Context& context, SemIR::EntityWithParamsBase& entity_base) -> void {
  if (!entity_base.implicit_param_patterns_id.has_value() ||
      entity_base.param_patterns_id.has_value()) {
    return;
  }

  CARBON_DIAGNOSTIC(GenericMissingExplicitParameters, Error,
                    "expected explicit parameters after implicit parameters");
  context.emitter().Emit(entity_base.last_param_node_id,
                         GenericMissingExplicitParameters);
}

}  // namespace Carbon::Check
