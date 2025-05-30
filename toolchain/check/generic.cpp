// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/generic.h"

#include <utility>

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

// Get the current pending generic. If we have not yet allocated a `GenericId`
// for it, do so now.
static auto GetOrCreatePendingGeneric(Context& context)
    -> GenericRegionStack::PendingGeneric {
  auto pending_generic = context.generic_region_stack().PeekPendingGeneric();
  if (!pending_generic.generic_id.has_value()) {
    // Allocate a placeholder generic now to form a generic ID. This generic
    // will be populated once we reach the end of the generic declaration.
    pending_generic.generic_id = context.generics().Add(
        SemIR::Generic{.decl_id = SemIR::InstId::None,
                       .bindings_id = SemIR::InstBlockId::None,
                       .self_specific_id = SemIR::SpecificId::None});
    context.generic_region_stack().SetPendingGenericId(
        pending_generic.generic_id);
  }
  return pending_generic;
}

// Adds an instruction `generic_inst_id` to the eval block for the current
// generic region. The instruction `generic_inst_id` is expected to compute the
// value of the constant described by `const_inst_id` in each specific. Forms
// and returns a corresponding symbolic constant ID that refers to the
// substituted value of that instruction in each specific.
static auto AddGenericConstantInstToEvalBlock(
    Context& context, SemIR::InstId const_inst_id,
    SemIR::InstId generic_inst_id, SemIR::ConstantDependence dependence)
    -> SemIR::ConstantId {
  auto [generic_id, region] = GetOrCreatePendingGeneric(context);
  auto index = SemIR::GenericInstIndex(
      region, context.generic_region_stack().PeekEvalBlock().size());
  context.generic_region_stack().AddInstToEvalBlock(generic_inst_id);
  return context.constant_values().AddSymbolicConstant(
      {.inst_id = const_inst_id,
       .generic_id = generic_id,
       .index = index,
       .dependence = dependence});
}

namespace {
// Substitution callbacks to rebuild a generic constant in the eval block for a
// generic region.
class RebuildGenericConstantInEvalBlockCallbacks : public SubstInstCallbacks {
 public:
  // `context` must not be null.
  RebuildGenericConstantInEvalBlockCallbacks(Context* context,
                                             SemIR::LocId loc_id)
      : SubstInstCallbacks(context),
        loc_id_(loc_id),
        constants_in_generic_(
            context->generic_region_stack().PeekConstantsInGenericMap()) {}

  auto RebuildType(SemIR::TypeInstId type_inst_id) const
      -> SemIR::TypeId override {
    // When building instructions in the eval block, form attached types.
    return context().types().GetTypeIdForTypeConstantId(
        context().constant_values().GetAttached(type_inst_id));
  }

  // Check for instructions for which we already have a mapping into the eval
  // block, and substitute them with the instructions in the eval block.
  auto Subst(SemIR::InstId& inst_id) const -> bool override {
    auto const_id = context().constant_values().Get(inst_id);
    if (!const_id.has_value()) {
      // An unloaded import ref should never contain anything we need to
      // substitute into. Don't trigger loading it here.
      CARBON_CHECK(
          context().insts().Is<SemIR::ImportRefUnloaded>(inst_id),
          "Substituting into instruction with invalid constant ID: {0}",
          context().insts().Get(inst_id));
      return true;
    }
    if (!context().constant_values().DependsOnGenericParameter(const_id)) {
      // This instruction doesn't have a symbolic constant value, so can't
      // contain any bindings that need to be substituted.
      return true;
    }

    // If this constant value has a defining instruction in the eval block,
    // replace the instruction in the body of the generic with the one from the
    // eval block.
    if (auto result = constants_in_generic_.Lookup(
            context().constant_values().GetInstId(const_id))) {
      inst_id = result.value();
      return true;
    }

    return false;
  }

  // Build a new instruction in the eval block corresponding to the given
  // constant.
  auto Rebuild(SemIR::InstId orig_inst_id, SemIR::Inst new_inst) const
      -> SemIR::InstId override {
    auto& orig_symbolic_const = context().constant_values().GetSymbolicConstant(
        context().constant_values().Get(orig_inst_id));
    auto const_inst_id = orig_symbolic_const.inst_id;
    auto dependence = orig_symbolic_const.dependence;

    // We might already have an instruction in the eval block if a transitive
    // operand of this instruction has the same constant value.
    auto result = constants_in_generic_.Insert(const_inst_id, [&] {
      // TODO: Add a function on `Context` to add the instruction without
      // inserting it into the dependent instructions list or computing a
      // constant value for it.
      // TODO: Is the location we pick here always appropriate for the new
      // instruction?
      auto inst_id = context().sem_ir().insts().AddInNoBlock(
          SemIR::LocIdAndInst::UncheckedLoc(loc_id_, new_inst));
      auto const_id = AddGenericConstantInstToEvalBlock(
          context(), const_inst_id, inst_id, dependence);
      context().constant_values().Set(inst_id, const_id);
      return inst_id;
    });
    return result.value();
  }

  auto ReuseUnchanged(SemIR::InstId orig_inst_id) const
      -> SemIR::InstId override {
    auto inst = context().insts().Get(orig_inst_id);
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
  SemIR::LocId loc_id_;
  ConstantsInGenericMap& constants_in_generic_;
};

// Substitution callbacks to rebuild a template action. This rebuilds the action
// instruction in-place if it needs to be modified.
class RebuildTemplateActionInEvalBlockCallbacks final
    : public RebuildGenericConstantInEvalBlockCallbacks {
 public:
  // `context` must not be null.
  RebuildTemplateActionInEvalBlockCallbacks(Context* context,
                                            SemIR::LocId loc_id,
                                            SemIR::InstId action_inst_id)
      : RebuildGenericConstantInEvalBlockCallbacks(context, loc_id),
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
// specific into the eval block for the current generic region. Returns a
// symbolic type ID that refers to the substituted type in each specific.
static auto AddGenericTypeToEvalBlock(Context& context, SemIR::LocId loc_id,
                                      SemIR::TypeId type_id) -> SemIR::TypeId {
  // Substitute into the type's constant instruction and rebuild it in the eval
  // block.
  auto type_inst_id =
      SubstInst(context, context.types().GetInstId(type_id),
                RebuildGenericConstantInEvalBlockCallbacks(&context, loc_id));
  return context.types().GetTypeIdForTypeConstantId(
      context.constant_values().GetAttached(type_inst_id));
}

// Adds instructions to compute the substituted value of `inst_id` in each
// specific into the eval block for the current generic region. Returns a
// symbolic constant instruction ID that refers to the substituted constant
// value in each specific.
static auto AddGenericConstantToEvalBlock(Context& context,
                                          SemIR::InstId inst_id)
    -> SemIR::ConstantId {
  CARBON_CHECK(context.constant_values().Get(inst_id).is_symbolic(),
               "Adding generic constant {0} with non-symbolic value {1}",
               context.insts().Get(inst_id),
               context.constant_values().Get(inst_id));

  // Substitute into the constant value and rebuild it in the eval block if
  // we've not encountered it before.
  auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);
  auto callbacks = RebuildGenericConstantInEvalBlockCallbacks(
      &context, SemIR::LocId(inst_id));
  auto new_inst_id = SubstInst(context, const_inst_id, callbacks);
  CARBON_CHECK(new_inst_id != const_inst_id,
               "No substitutions performed for generic constant {0}",
               context.insts().Get(inst_id));
  return context.constant_values().GetAttached(new_inst_id);
}

// Adds an instruction that performs a template action to the eval block for the
// generic. The instruction should not yet have been added to any block. The
// instruction might refer to types and constants that need to be rewritten, so
// substitute into it first.
static auto AddTemplateActionToEvalBlock(Context& context,
                                         SemIR::InstId inst_id) -> void {
  // Substitute into the constant value and rebuild it in the eval block.
  auto new_inst_id = SubstInst(context, inst_id,
                               RebuildTemplateActionInEvalBlockCallbacks(
                                   &context, SemIR::LocId(inst_id), inst_id));
  CARBON_CHECK(new_inst_id == inst_id,
               "Substitution changed InstId of template action");
  context.generic_region_stack().PeekConstantsInGenericMap().Insert(inst_id,
                                                                    inst_id);

  // Add the action to the eval block and point its constant value back to its
  // index within the block.
  auto [generic_id, region] = GetOrCreatePendingGeneric(context);
  auto& symbolic_constant = context.constant_values().GetSymbolicConstant(
      context.constant_values().GetAttached(inst_id));
  symbolic_constant.generic_id = generic_id;
  symbolic_constant.index = SemIR::GenericInstIndex(
      region, context.generic_region_stack().PeekEvalBlock().size());
  context.generic_region_stack().AddInstToEvalBlock(inst_id);
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

auto AttachDependentInstToCurrentGeneric(Context& context,
                                         DependentInst dependent_inst) -> void {
  auto [inst_id, dep_kind] = dependent_inst;

  // If we don't have a generic region here, leave the dependent instruction
  // unattached. This happens for out-of-line redeclarations of members of
  // dependent scopes:
  //
  //   class A(T:! type) {
  //     fn F();
  //   }
  //   // Has generic type and constant value, but no generic region.
  //   fn A(T:! type).F() {}
  //
  // TODO: Copy the attached type and constant value from the previous
  // declaration in this case instead of attempting to attach the new
  // declaration to a generic region that we're no longer within.
  if (context.generic_region_stack().Empty()) {
    // This should only happen for `*Decl` instructions, never for template
    // actions.
    CARBON_CHECK((dep_kind & DependentInst::Template) == DependentInst::None);
    return;
  }

  context.generic_region_stack().AddDependentInst(dependent_inst.inst_id);

  // If the type is symbolic, replace it with a type specific to this generic.
  if ((dep_kind & DependentInst::SymbolicType) != DependentInst::None) {
    auto inst = context.insts().Get(inst_id);
    auto type_id = AddGenericTypeToEvalBlock(context, SemIR::LocId(inst_id),
                                             inst.type_id());
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
  if ((dep_kind & DependentInst::SymbolicConstant) != DependentInst::None) {
    // Update the constant value to refer to this generic.
    context.constant_values().Set(
        inst_id, AddGenericConstantToEvalBlock(context, inst_id));
  }

  // If the instruction is a template action, add it directly to this position
  // in the eval block.
  if ((dep_kind & DependentInst::Template) != DependentInst::None) {
    AddTemplateActionToEvalBlock(context, inst_id);
  }
}

// Builds and returns a block of instructions whose constant values need to be
// evaluated in order to resolve a generic to a specific.
static auto MakeGenericEvalBlock(Context& context) -> SemIR::InstBlockId {
  return context.inst_blocks().Add(
      context.generic_region_stack().PeekEvalBlock());
}

// Builds and returns an eval block, given the list of canonical symbolic
// constants that the instructions in the eval block should produce. This is
// used when importing a generic.
auto RebuildGenericEvalBlock(Context& context, SemIR::GenericId generic_id,
                             SemIR::GenericInstIndex::Region region,
                             llvm::ArrayRef<SemIR::InstId> const_ids)
    -> SemIR::InstBlockId {
  context.generic_region_stack().Push(
      {.generic_id = generic_id, .region = region});

  auto& constants_in_generic =
      context.generic_region_stack().PeekConstantsInGenericMap();

  // For the definition region, populate constants from the declaration.
  if (region == SemIR::GenericInstIndex::Definition) {
    PopulateConstantsFromDeclaration(context, generic_id, constants_in_generic);
  }

  constants_in_generic.GrowForInsertCount(const_ids.size());
  for (auto [i, inst_id] : llvm::enumerate(const_ids)) {
    // Build a constant in the inst block.
    AddGenericConstantToEvalBlock(context, inst_id);
    CARBON_CHECK(context.generic_region_stack().PeekEvalBlock().size() == i + 1,
                 "Produced {0} instructions when importing {1}",
                 (context.generic_region_stack().PeekEvalBlock().size() - i),
                 context.insts().Get(inst_id));
  }

  auto eval_block_id = MakeGenericEvalBlock(context);
  context.generic_region_stack().Pop();
  return eval_block_id;
}

auto StartGenericDecl(Context& context) -> void {
  context.generic_region_stack().Push(
      {.generic_id = SemIR::GenericId::None,
       .region = SemIR::GenericInstIndex::Declaration});
}

auto StartGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void {
  // Push a generic region even if we don't have a generic_id. We might still
  // have locally-introduced generic parameters to track:
  //
  // fn F() {
  //   let T:! type = i32;
  //   var x: T;
  // }
  context.generic_region_stack().Push(
      {.generic_id = generic_id,
       .region = SemIR::GenericInstIndex::Definition});
  if (generic_id.has_value()) {
    PopulateConstantsFromDeclaration(
        context, generic_id,
        context.generic_region_stack().PeekConstantsInGenericMap());
  }
}

auto DiscardGenericDecl(Context& context) -> void {
  // Unattach any types and constant values we might have created in the
  // generic.
  // TODO: We should re-evaluate the contents of the eval block in a synthesized
  // specific to form these values, in order to propagate the values of local
  // `let :!` bindings.
  for (auto inst_id : context.generic_region_stack().PeekDependentInsts()) {
    // Note that `Get` returns an instruction with an unattached type.
    context.sem_ir().insts().Set(inst_id, context.insts().Get(inst_id));
    // Note that `Get` returns an unattached constant.
    context.constant_values().Set(inst_id,
                                  context.constant_values().Get(inst_id));
  }
  // Note that we may leak a GenericId here, if one was allocated.
  context.generic_region_stack().Pop();
}

auto BuildGeneric(Context& context, SemIR::InstId decl_id) -> SemIR::GenericId {
  auto all_bindings =
      context.scope_stack().compile_time_bindings_stack().PeekAllValues();

  if (all_bindings.empty()) {
    CARBON_CHECK(context.generic_region_stack().PeekEvalBlock().empty(),
                 "Have non-empty eval block {0} in declaration {1} but no "
                 "compile time bindings are in scope.",
                 context.insts().Get(
                     context.generic_region_stack().PeekEvalBlock().front()),
                 context.insts().Get(decl_id));
    DiscardGenericDecl(context);
    return SemIR::GenericId::None;
  }

  // Build the new Generic object. Note that we intentionally do not hold a
  // persistent reference to it throughout this function, because the `generics`
  // collection can have items added to it by import resolution while we are
  // building this generic.
  auto bindings_id = context.inst_blocks().Add(all_bindings);

  SemIR::Generic generic = {.decl_id = decl_id,
                            .bindings_id = bindings_id,
                            .self_specific_id = SemIR::SpecificId::None};

  // Get the generic ID, or allocate one now if we don't have one yet. That
  // could happen if the eval block is empty.
  auto generic_id =
      context.generic_region_stack().PeekPendingGeneric().generic_id;
  if (!generic_id.has_value()) {
    CARBON_CHECK(context.generic_region_stack().PeekEvalBlock().empty(),
                 "Non-empty eval block but didn't yet allocate a GenericId");
    generic_id = context.generics().Add(generic);
    context.generic_region_stack().SetPendingGenericId(generic_id);
  } else {
    CARBON_CHECK(!context.generics().Get(generic_id).decl_id.has_value(),
                 "Built generic {0} twice", generic_id);
    context.generics().Get(generic_id) = generic;
  }

  // MakeSelfSpecificId could cause something to be imported, which would
  // invalidate the return value of `context.generics().Get(generic_id)`.
  auto self_specific_id = MakeSelfSpecificId(context, generic_id);
  context.generics().Get(generic_id).self_specific_id = self_specific_id;
  return generic_id;
}

auto FinishGenericDecl(Context& context, SemIR::LocId loc_id,
                       SemIR::GenericId generic_id) -> void {
  if (!generic_id.has_value()) {
    return;
  }
  auto decl_block_id = MakeGenericEvalBlock(context);
  context.generic_region_stack().Pop();
  context.generics().Get(generic_id).decl_block_id = decl_block_id;

  ResolveSpecificDecl(context, loc_id,
                      context.generics().GetSelfSpecific(generic_id));
}

auto BuildGenericDecl(Context& context, SemIR::InstId decl_id)
    -> SemIR::GenericId {
  SemIR::GenericId generic_id = BuildGeneric(context, decl_id);
  if (generic_id.has_value()) {
    FinishGenericDecl(context, SemIR::LocId(decl_id), generic_id);
  }
  return generic_id;
}

// Returns the first difference between the two given eval blocks.
static auto FirstDifferenceBetweenEvalBlocks(
    Context& context, llvm::ArrayRef<SemIR::InstId> old_eval_block,
    llvm::ArrayRef<SemIR::InstId> new_eval_block)
    -> std::pair<SemIR::InstId, SemIR::InstId> {
  // Check each element of the eval block computes the same unattached constant.
  for (auto [old_inst_id, new_inst_id] :
       llvm::zip(old_eval_block, new_eval_block)) {
    auto old_const_id = context.constant_values().Get(old_inst_id);
    auto new_const_id = context.constant_values().Get(new_inst_id);
    if (old_const_id != new_const_id) {
      if (old_const_id.is_symbolic() && new_const_id.is_symbolic() &&
          context.constant_values().GetDependence(old_const_id) ==
              SemIR::ConstantDependence::Template &&
          context.constant_values().GetDependence(new_const_id) ==
              SemIR::ConstantDependence::Template &&
          context.insts().Get(old_inst_id).kind() ==
              context.insts().Get(new_inst_id).kind()) {
        // TODO: We don't have a good mechanism to compare template constants
        // because they canonicalize to themselves, so just assume this is OK.
        continue;
      }

      // These constant values differ unexpectedly.
      return {old_inst_id, new_inst_id};
    }
  }

  if (old_eval_block.size() < new_eval_block.size()) {
    return {SemIR::InstId::None, new_eval_block[old_eval_block.size()]};
  }
  if (old_eval_block.size() > new_eval_block.size()) {
    return {old_eval_block[new_eval_block.size()], SemIR::InstId::None};
  }

  return {SemIR::InstId::None, SemIR::InstId::None};
}

// If `constant_id` refers to a symbolic constant within the declaration region
// of `generic_id`, remap it to refer to the constant value of the corresponding
// element in the given eval block. Otherwise returns the ID unchanged.
static auto ReattachConstant(Context& context, SemIR::GenericId generic_id,
                             llvm::ArrayRef<SemIR::InstId> eval_block,
                             SemIR::ConstantId constant_id)
    -> SemIR::ConstantId {
  if (!constant_id.has_value() || !constant_id.is_symbolic()) {
    return constant_id;
  }

  auto& symbolic_const =
      context.constant_values().GetSymbolicConstant(constant_id);
  if (symbolic_const.generic_id != generic_id) {
    // Constant doesn't refer into this generic.
    return constant_id;
  }

  CARBON_CHECK(
      symbolic_const.index.region() == SemIR::GenericInstIndex::Declaration,
      "Definition region of redeclaration should not be referenced");
  return context.constant_values().GetAttached(
      eval_block[symbolic_const.index.index()]);
}

// Same as `ReattachConstant` but for a type.
static auto ReattachType(Context& context, SemIR::GenericId generic_id,
                         llvm::ArrayRef<SemIR::InstId> eval_block,
                         SemIR::TypeId type_id) -> SemIR::TypeId {
  return context.types().GetTypeIdForTypeConstantId(ReattachConstant(
      context, generic_id, eval_block, context.types().GetConstantId(type_id)));
}

auto FinishGenericRedecl(Context& context, SemIR::GenericId generic_id)
    -> void {
  if (!generic_id.has_value()) {
    DiscardGenericDecl(context);
    return;
  }

  // Find the old and new eval blocks.
  auto old_eval_block_id =
      context.generics()
          .Get(generic_id)
          .GetEvalBlock(SemIR::GenericInstIndex::Declaration);
  CARBON_CHECK(old_eval_block_id.has_value(),
               "Old generic is not fully declared");

  auto old_eval_block = context.inst_blocks().Get(old_eval_block_id);
  auto new_eval_block = context.generic_region_stack().PeekEvalBlock();

  // Check the eval blocks are computing the same constants in the same order.
  // This should always be the case because we have already verified they have
  // the same parse tree, and the poisoning rules mean that all entities they
  // refer to are also the same.
  //
  // Note that it's OK if the first difference is that an old instruction has no
  // corresponding new instruction; we wouldn't have used that anyway. This
  // happens for `ImplDecl`, for which the witness is included in the eval block
  // of the first declaration.
  if (auto [old_inst_id, new_inst_id] = FirstDifferenceBetweenEvalBlocks(
          context, old_eval_block, new_eval_block);
      new_inst_id.has_value()) {
    // This shouldn't be possible: we should have already checked that the
    // syntax of the redeclaration matches the prior declaration, and none of
    // the name lookups or semantic checks should be allowed to differ between
    // the two declarations, so we should have built the same eval block as in
    // the prior declaration.
    //
    // However, that isn't a strong enough invariant that it seems appropriate
    // to CHECK-fail here, so we produce a diagnostic with context.TODO()
    // instead.
    //
    // TODO: Add something like context.UNEXPECTED() instead of using
    // context.TODO() here because there's not really anything to do.
    context.TODO(new_inst_id,
                 "generic redeclaration differs from previous declaration");
    if (old_inst_id.has_value()) {
      context.TODO(old_inst_id, "instruction in previous declaration");
    }
    DiscardGenericDecl(context);
    return;
  }

  auto redecl_generic_id =
      context.generic_region_stack().PeekPendingGeneric().generic_id;

  // Reattach any instructions that depend on the redeclaration to instead refer
  // to the original.
  for (auto inst_id : context.generic_region_stack().PeekDependentInsts()) {
    // Reattach the type.
    auto inst = context.insts().GetWithAttachedType(inst_id);
    inst.SetType(ReattachType(context, redecl_generic_id, old_eval_block,
                              inst.type_id()));
    context.sem_ir().insts().Set(inst_id, inst);

    // Reattach the constant value.
    context.constant_values().Set(
        inst_id,
        ReattachConstant(context, redecl_generic_id, old_eval_block,
                         context.constant_values().GetAttached(inst_id)));
  }
  context.generic_region_stack().Pop();
}

auto FinishGenericDefinition(Context& context, SemIR::GenericId generic_id)
    -> void {
  if (!generic_id.has_value()) {
    DiscardGenericDecl(context);
    return;
  }

  auto definition_block_id = MakeGenericEvalBlock(context);
  context.generic_region_stack().Pop();
  context.generics().Get(generic_id).definition_block_id = definition_block_id;
}

auto ResolveSpecificDecl(Context& context, SemIR::LocId loc_id,
                         SemIR::SpecificId specific_id) -> void {
  // If this is the first time we've formed this specific, evaluate its decl
  // block to form information about the specific.
  if (!context.specifics().Get(specific_id).decl_block_id.has_value()) {
    // Set a placeholder value as the decl block ID so we won't attempt to
    // recursively resolve the same specific.
    context.specifics().Get(specific_id).decl_block_id =
        SemIR::InstBlockId::Empty;

    auto decl_block_id =
        TryEvalBlockForSpecific(context, loc_id, specific_id,
                                SemIR::GenericInstIndex::Region::Declaration);
    // Note that TryEvalBlockForSpecific may reallocate the list of specifics,
    // so re-lookup the specific here.
    context.specifics().Get(specific_id).decl_block_id = decl_block_id;
  }
}

auto MakeSpecific(Context& context, SemIR::LocId loc_id,
                  SemIR::GenericId generic_id, SemIR::InstBlockId args_id)
    -> SemIR::SpecificId {
  auto specific_id = context.specifics().GetOrAdd(generic_id, args_id);
  ResolveSpecificDecl(context, loc_id, specific_id);
  return specific_id;
}

auto MakeSpecific(Context& context, SemIR::LocId loc_id,
                  SemIR::GenericId generic_id,
                  llvm::ArrayRef<SemIR::InstId> args) -> SemIR::SpecificId {
  auto args_id = context.inst_blocks().AddCanonical(args);
  return MakeSpecific(context, loc_id, generic_id, args_id);
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

auto MakeSelfSpecific(Context& context, SemIR::LocId loc_id,
                      SemIR::GenericId generic_id) -> SemIR::SpecificId {
  // Build a corresponding specific.
  SemIR::SpecificId specific_id = MakeSelfSpecificId(context, generic_id);
  // TODO: This could be made more efficient. We don't need to perform
  // substitution here; we know we want identity mappings for all constants and
  // types. We could also consider not storing the mapping at all in this case.
  ResolveSpecificDecl(context, loc_id, specific_id);
  return specific_id;
}

auto ResolveSpecificDefinition(Context& context, SemIR::LocId loc_id,
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
        context, loc_id, specific_id, SemIR::GenericInstIndex::Definition);
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
