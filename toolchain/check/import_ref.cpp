// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_ref.h"

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Resolves an instruction from an imported IR into a constant referring to the
// current IR.
//
// Calling Resolve on an instruction operates in an iterative manner, tracking
// Work items on work_stack_. At a high level, the loop is:
//
// 1. If a constant value is already known for the work item, and we're
//    processing it for the first time, it's considered resolved.
//    - The constant check avoids performance costs of deduplication on add.
//    - If `retry` is set, we process it again, because it didn't complete last
//      time, even though we have a constant value already.
// 2. Resolve the instruction: (TryResolveInst/TryResolveTypedInst)
//    - For instructions that can be forward declared, if we don't already have
//      a constant value from a previous attempt at resolution, start by making
//      a forward declared constant value to address circular references.
//    - Gather all input constants.
//      - Gathering constants directly adds unresolved values to work_stack_.
//    - If any need to be resolved (HasNewWork), return Retry(): this
//      instruction needs two calls to complete.
//      - If the constant value is already known because we have made a forward
//        declaration, pass it to Retry(). It will be passed to future attempts
//        to resolve this instruction so the earlier work can be found, and will
//        be made available for other instructions to use.
//      - The second attempt to resolve this instruction must produce the same
//        constant, because the value may have already been used by resolved
//        instructions.
//    - Build any necessary IR structures, and return the output constant.
// 3. If resolve didn't return Retry(), pop the work. Otherwise, it needs to
//    remain, and may no longer be at the top of the stack; set `retry` on it so
//    we'll make sure to run it again later.
//
// TryResolveInst/TryResolveTypedInst can complete in one call for a given
// instruction, but should always complete within two calls. However, due to the
// chance of a second call, it's important to reserve all expensive logic until
// it's been established that input constants are available; this in particular
// includes GetTypeIdForTypeConstant calls which do a hash table lookup.
class ImportRefResolver {
 public:
  explicit ImportRefResolver(Context& context, SemIR::ImportIRId import_ir_id)
      : context_(context),
        import_ir_id_(import_ir_id),
        import_ir_(*context_.import_irs().Get(import_ir_id).sem_ir),
        import_ir_constant_values_(
            context_.import_ir_constant_values()[import_ir_id.index]) {}

  // Iteratively resolves an imported instruction's inner references until a
  // constant ID referencing the current IR is produced. See the class comment
  // for more details.
  auto Resolve(SemIR::InstId inst_id) -> SemIR::ConstantId {
    work_stack_.push_back({inst_id});
    while (!work_stack_.empty()) {
      auto work = work_stack_.back();
      CARBON_CHECK(work.inst_id.is_valid());

      // Step 1: check for a constant value.
      auto existing_const_id = import_ir_constant_values_.Get(work.inst_id);
      if (existing_const_id.is_valid() && !work.retry) {
        work_stack_.pop_back();
        continue;
      }

      // Step 2: resolve the instruction.
      auto initial_work = work_stack_.size();
      auto [new_const_id, finished] =
          TryResolveInst(work.inst_id, existing_const_id);
      CARBON_CHECK(finished == !HasNewWork(initial_work));

      CARBON_CHECK(!existing_const_id.is_valid() ||
                   existing_const_id == new_const_id)
          << "Constant value changed in second pass.";
      import_ir_constant_values_.Set(work.inst_id, new_const_id);

      // Step 3: pop or retry.
      if (finished) {
        work_stack_.pop_back();
      } else {
        work_stack_[initial_work - 1].retry = true;
      }
    }
    auto constant_id = import_ir_constant_values_.Get(inst_id);
    CARBON_CHECK(constant_id.is_valid());
    return constant_id;
  }

  // Wraps constant evaluation with logic to handle types.
  auto ResolveType(SemIR::TypeId import_type_id) -> SemIR::TypeId {
    if (!import_type_id.is_valid()) {
      return import_type_id;
    }

    auto import_type_inst_id = import_ir_.types().GetInstId(import_type_id);
    CARBON_CHECK(import_type_inst_id.is_valid());

    if (import_type_inst_id.is_builtin()) {
      // Builtins don't require constant resolution; we can use them directly.
      return context_.GetBuiltinType(import_type_inst_id.builtin_kind());
    } else {
      return context_.GetTypeIdForTypeConstant(Resolve(import_type_inst_id));
    }
  }

 private:
  // A step in work_stack_.
  struct Work {
    // The instruction to work on.
    SemIR::InstId inst_id;

    // True if another pass was requested last time this was run.
    bool retry = false;
  };

  // The result of attempting to resolve an imported instruction to a constant.
  struct ResolveResult {
    // Try resolving this function again. If `const_id` is specified, it will be
    // passed to the next resolution attempt.
    static auto Retry(SemIR::ConstantId const_id = SemIR::ConstantId::Invalid)
        -> ResolveResult {
      return {.const_id = const_id, .finished = false};
    }

    // The new constant value, if known.
    SemIR::ConstantId const_id;
    // Whether resolution has finished. If false, `TryResolveInst` will be
    // called again. Note that this is not strictly necessary, and we can get
    // the same information by checking whether new work was added to the stack.
    // However, we use this for consistency checks between resolve actions and
    // the work stack.
    bool finished = true;
  };

  // Returns true if new unresolved constants were found.
  //
  // At the start of a function, do:
  //   auto initial_work = work_stack_.size();
  // Then when determining:
  //   if (HasNewWork(initial_work)) { ... }
  auto HasNewWork(size_t initial_work) -> bool {
    CARBON_CHECK(initial_work <= work_stack_.size())
        << "Work shouldn't decrease";
    return initial_work < work_stack_.size();
  }

  auto AddImportIRInst(SemIR::InstId inst_id) -> SemIR::LocId {
    return context_.import_ir_insts().Add(
        {.ir_id = import_ir_id_, .inst_id = inst_id});
  }

  // Returns the ConstantId for an InstId. Adds unresolved constants to
  // work_stack_.
  auto GetLocalConstantId(SemIR::InstId inst_id) -> SemIR::ConstantId {
    auto const_id = import_ir_constant_values_.Get(inst_id);
    if (!const_id.is_valid()) {
      work_stack_.push_back({inst_id});
    }
    return const_id;
  }

  // Returns the ConstantId for a TypeId. Adds unresolved constants to
  // work_stack_.
  auto GetLocalConstantId(SemIR::TypeId type_id) -> SemIR::ConstantId {
    return GetLocalConstantId(import_ir_.types().GetInstId(type_id));
  }

  // Returns the ConstantId for each parameter's type. Adds unresolved constants
  // to work_stack_.
  auto GetLocalParamConstantIds(SemIR::InstBlockId param_refs_id)
      -> llvm::SmallVector<SemIR::ConstantId> {
    if (param_refs_id == SemIR::InstBlockId::Empty) {
      return {};
    }
    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> const_ids;
    const_ids.reserve(param_refs.size());
    for (auto inst_id : param_refs) {
      const_ids.push_back(
          GetLocalConstantId(import_ir_.insts().Get(inst_id).type_id()));

      // If the parameter is a symbolic binding, build the BindSymbolicName
      // constant.
      auto bind_id = inst_id;
      if (auto addr =
              import_ir_.insts().TryGetAs<SemIR::AddrPattern>(bind_id)) {
        bind_id = addr->inner_id;
      }
      GetLocalConstantId(bind_id);
    }
    return const_ids;
  }

  // Given a param_refs_id and const_ids from GetLocalParamConstantIds, returns
  // a version of param_refs_id localized to the current IR.
  auto GetLocalParamRefsId(
      SemIR::InstBlockId param_refs_id,
      const llvm::SmallVector<SemIR::ConstantId>& const_ids)
      -> SemIR::InstBlockId {
    if (param_refs_id == SemIR::InstBlockId::Empty) {
      return SemIR::InstBlockId::Empty;
    }
    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
    llvm::SmallVector<SemIR::InstId> new_param_refs;
    for (auto [ref_id, const_id] : llvm::zip(param_refs, const_ids)) {
      // Figure out the param structure. This echoes
      // Function::GetParamFromParamRefId.
      // TODO: Consider a different parameter handling to simplify import logic.
      auto inst = import_ir_.insts().Get(ref_id);
      auto addr_inst = inst.TryAs<SemIR::AddrPattern>();

      auto bind_id = ref_id;
      auto param_id = ref_id;

      if (addr_inst) {
        bind_id = addr_inst->inner_id;
        param_id = bind_id;
        inst = import_ir_.insts().Get(bind_id);
      }

      auto bind_inst = inst.TryAs<SemIR::AnyBindName>();
      if (bind_inst) {
        param_id = bind_inst->value_id;
        inst = import_ir_.insts().Get(param_id);
      }
      auto param_inst = inst.As<SemIR::Param>();

      // Rebuild the param instruction.
      auto name_id = GetLocalNameId(param_inst.name_id);
      auto type_id = context_.GetTypeIdForTypeConstant(const_id);

      auto new_param_id = context_.AddInstInNoBlock(
          {AddImportIRInst(param_id), SemIR::Param{type_id, name_id}});
      if (bind_inst) {
        switch (bind_inst->kind) {
          case SemIR::BindName::Kind: {
            auto bind_name_id = context_.bind_names().Add(
                {.name_id = name_id,
                 .enclosing_scope_id = SemIR::NameScopeId::Invalid});
            new_param_id = context_.AddInstInNoBlock(
                {AddImportIRInst(bind_id),
                 SemIR::BindName{type_id, bind_name_id, new_param_id}});
            break;
          }
          case SemIR::BindSymbolicName::Kind: {
            // The symbolic name will be created on first reference, so might
            // already exist. Update the value in it to refer to the parameter.
            auto new_bind_inst =
                context_.insts().GetAs<SemIR::BindSymbolicName>(
                    GetLocalConstantId(bind_id).inst_id());
            new_bind_inst.value_id = new_param_id;
            // This is not before constant use, but doesn't change the
            // constant value of the instruction.
            context_.ReplaceInstBeforeConstantUse(bind_id, new_bind_inst);
            break;
          }
          default: {
            CARBON_FATAL() << "Unexpected kind: " << bind_inst->kind;
          }
        }
      }
      if (addr_inst) {
        new_param_id = context_.AddInstInNoBlock(SemIR::LocIdAndInst::Untyped(
            AddImportIRInst(ref_id),
            SemIR::AddrPattern{type_id, new_param_id}));
      }
      new_param_refs.push_back(new_param_id);
    }
    return context_.inst_blocks().Add(new_param_refs);
  }

  // Translates a NameId from the import IR to a local NameId.
  auto GetLocalNameId(SemIR::NameId import_name_id) -> SemIR::NameId {
    if (auto ident_id = import_name_id.AsIdentifierId(); ident_id.is_valid()) {
      return SemIR::NameId::ForIdentifier(
          context_.identifiers().Add(import_ir_.identifiers().Get(ident_id)));
    }
    return import_name_id;
  }

  // Translates a NameScopeId from the import IR to a local NameScopeId. Adds
  // unresolved constants to the work stack.
  auto GetLocalNameScopeId(SemIR::NameScopeId name_scope_id)
      -> SemIR::NameScopeId {
    auto inst_id = import_ir_.name_scopes().GetInstIdIfValid(name_scope_id);
    if (!inst_id.is_valid()) {
      // Map scopes that aren't associated with an instruction to invalid
      // scopes. For now, such scopes aren't used, and we don't have a good way
      // to remap them.
      return SemIR::NameScopeId::Invalid;
    }
    if (import_ir_.insts().Is<SemIR::ImplDecl>(inst_id)) {
      // TODO: Import the scope for an `impl` definition.
      return SemIR::NameScopeId::Invalid;
    }
    auto const_id = GetLocalConstantId(inst_id);
    if (!const_id.is_valid()) {
      return SemIR::NameScopeId::Invalid;
    }
    auto name_scope_inst = context_.insts().Get(const_id.inst_id());
    CARBON_KIND_SWITCH(name_scope_inst) {
      case CARBON_KIND(SemIR::Namespace inst): {
        return inst.name_scope_id;
      }
      case CARBON_KIND(SemIR::ClassType inst): {
        return context_.classes().Get(inst.class_id).scope_id;
      }
      case CARBON_KIND(SemIR::InterfaceType inst): {
        return context_.interfaces().Get(inst.interface_id).scope_id;
      }
      default:
        if (const_id == SemIR::ConstantId::Error) {
          return SemIR::NameScopeId::Invalid;
        }
        CARBON_FATAL() << "Unexpected instruction kind for name scope: "
                       << name_scope_inst;
    }
  }

  // Adds ImportRefUnloaded entries for members of the imported scope, for name
  // lookup.
  auto AddNameScopeImportRefs(const SemIR::NameScope& import_scope,
                              SemIR::NameScope& new_scope) -> void {
    for (auto [entry_name_id, entry_inst_id] : import_scope.names) {
      auto ref_id = context_.AddImportRef(
          {.ir_id = import_ir_id_, .inst_id = entry_inst_id});
      CARBON_CHECK(
          new_scope.names.insert({GetLocalNameId(entry_name_id), ref_id})
              .second);
    }
  }

  // Given a block ID for a list of associated entities of a witness, returns a
  // version localized to the current IR.
  auto AddAssociatedEntities(SemIR::InstBlockId associated_entities_id)
      -> SemIR::InstBlockId {
    if (associated_entities_id == SemIR::InstBlockId::Empty) {
      return SemIR::InstBlockId::Empty;
    }
    auto associated_entities =
        import_ir_.inst_blocks().Get(associated_entities_id);
    llvm::SmallVector<SemIR::InstId> new_associated_entities;
    new_associated_entities.reserve(associated_entities.size());
    for (auto inst_id : associated_entities) {
      new_associated_entities.push_back(
          context_.AddImportRef({.ir_id = import_ir_id_, .inst_id = inst_id}));
    }
    return context_.inst_blocks().Add(new_associated_entities);
  }

  // Tries to resolve the InstId, returning a constant when ready, or Invalid if
  // more has been added to the stack. A similar API is followed for all
  // following TryResolveTypedInst helper functions.
  //
  // `const_id` is Invalid unless we've tried to resolve this instruction
  // before, in which case it's the previous result.
  //
  // TODO: Error is returned when support is missing, but that should go away.
  auto TryResolveInst(SemIR::InstId inst_id, SemIR::ConstantId const_id)
      -> ResolveResult {
    if (inst_id.is_builtin()) {
      CARBON_CHECK(!const_id.is_valid());
      // Constants for builtins can be directly copied.
      return {context_.constant_values().Get(inst_id)};
    }

    auto untyped_inst = import_ir_.insts().Get(inst_id);
    CARBON_KIND_SWITCH(untyped_inst) {
      case CARBON_KIND(SemIR::AssociatedEntity inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::AssociatedEntityType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::BaseDecl inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::BindAlias inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::BindName inst): {
        // TODO: This always returns `ConstantId::NotConstant`.
        return {TryEvalInst(context_, inst_id, inst)};
      }
      case CARBON_KIND(SemIR::BindSymbolicName inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::ClassDecl inst): {
        return TryResolveTypedInst(inst, const_id);
      }
      case CARBON_KIND(SemIR::ClassType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::ConstType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::FieldDecl inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::FunctionDecl inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::ImportRefUsed inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::InterfaceDecl inst): {
        return TryResolveTypedInst(inst, const_id);
      }
      case CARBON_KIND(SemIR::InterfaceWitness inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::InterfaceType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::PointerType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::StructType inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::TupleType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::UnboundElementType inst): {
        return TryResolveTypedInst(inst);
      }
      default:
        context_.TODO(
            AddImportIRInst(inst_id),
            llvm::formatv("TryResolveInst on {0}", untyped_inst.kind()).str());
        return {SemIR::ConstantId::Error};
    }
  }

  auto TryResolveTypedInst(SemIR::AssociatedEntity inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Add a lazy reference to the target declaration.
    auto decl_id = context_.AddImportRef(
        {.ir_id = import_ir_id_, .inst_id = inst.decl_id});

    auto inst_id = context_.AddInstInNoBlock(
        {AddImportIRInst(inst.decl_id),
         SemIR::AssociatedEntity{
             context_.GetTypeIdForTypeConstant(type_const_id), inst.index,
             decl_id}});
    return {context_.constant_values().Get(inst_id)};
  }

  auto TryResolveTypedInst(SemIR::AssociatedEntityType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    auto initial_work = work_stack_.size();
    auto entity_type_const_id = GetLocalConstantId(inst.entity_type_id);
    auto interface_const_id = GetLocalConstantId(
        import_ir_.interfaces().Get(inst.interface_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    auto inst_id = context_.AddInstInNoBlock(SemIR::AssociatedEntityType{
        SemIR::TypeId::TypeType,
        context_.insts()
            .GetAs<SemIR::InterfaceType>(interface_const_id.inst_id())
            .interface_id,
        context_.GetTypeIdForTypeConstant(entity_type_const_id)});
    return {context_.constant_values().Get(inst_id)};
  }

  auto TryResolveTypedInst(SemIR::BaseDecl inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);
    auto base_type_const_id = GetLocalConstantId(inst.base_type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Import the instruction in order to update contained base_type_id.
    auto inst_id = context_.AddInstInNoBlock(SemIR::LocIdAndInst::Untyped(
        AddImportIRInst(import_inst_id),
        SemIR::BaseDecl{context_.GetTypeIdForTypeConstant(type_const_id),
                        context_.GetTypeIdForTypeConstant(base_type_const_id),
                        inst.index}));
    return {context_.constant_values().Get(inst_id)};
  }

  auto TryResolveTypedInst(SemIR::BindAlias inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto value_id = GetLocalConstantId(inst.value_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    return {value_id};
  }

  auto TryResolveTypedInst(SemIR::BindSymbolicName inst,
                           SemIR::InstId import_inst_id) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    auto name_id =
        GetLocalNameId(import_ir_.bind_names().Get(inst.bind_name_id).name_id);
    auto bind_name_id = context_.bind_names().Add(
        {.name_id = name_id,
         .enclosing_scope_id = SemIR::NameScopeId::Invalid});
    auto new_bind_id = context_.AddInstInNoBlock(
        {AddImportIRInst(import_inst_id),
         SemIR::BindSymbolicName{context_.GetTypeIdForTypeConstant(type_id),
                                 bind_name_id, SemIR::InstId::Invalid}});
    return {context_.constant_values().Get(new_bind_id)};
  }

  // Makes an incomplete class. This is necessary even with classes with a
  // complete declaration, because things such as `Self` may refer back to the
  // type.
  auto MakeIncompleteClass(const SemIR::Class& import_class)
      -> SemIR::ConstantId {
    auto class_decl =
        SemIR::ClassDecl{SemIR::TypeId::Invalid, SemIR::ClassId::Invalid,
                         SemIR::InstBlockId::Empty};
    auto class_decl_id =
        context_.AddPlaceholderInst(SemIR::LocIdAndInst::Untyped(
            AddImportIRInst(import_class.decl_id), class_decl));
    // Regardless of whether ClassDecl is a complete type, we first need an
    // incomplete type so that any references have something to point at.
    class_decl.class_id = context_.classes().Add({
        .name_id = GetLocalNameId(import_class.name_id),
        // Set in the second pass once we've imported it.
        .enclosing_scope_id = SemIR::NameScopeId::Invalid,
        // `.self_type_id` depends on the ClassType, so is set below.
        .self_type_id = SemIR::TypeId::Invalid,
        .decl_id = class_decl_id,
        .inheritance_kind = import_class.inheritance_kind,
    });

    // Write the class ID into the ClassDecl.
    context_.ReplaceInstBeforeConstantUse(class_decl_id, class_decl);
    auto self_const_id = context_.constant_values().Get(class_decl_id);

    // Build the `Self` type using the resulting type constant.
    auto& class_info = context_.classes().Get(class_decl.class_id);
    class_info.self_type_id = context_.GetTypeIdForTypeConstant(self_const_id);
    return self_const_id;
  }

  // Fills out the class definition for an incomplete class.
  auto AddClassDefinition(const SemIR::Class& import_class,
                          SemIR::Class& new_class,
                          SemIR::ConstantId object_repr_const_id,
                          SemIR::ConstantId base_const_id) -> void {
    new_class.definition_id = new_class.decl_id;

    new_class.object_repr_id =
        context_.GetTypeIdForTypeConstant(object_repr_const_id);

    new_class.scope_id =
        context_.name_scopes().Add(new_class.decl_id, SemIR::NameId::Invalid,
                                   new_class.enclosing_scope_id);
    auto& new_scope = context_.name_scopes().Get(new_class.scope_id);
    const auto& import_scope =
        import_ir_.name_scopes().Get(import_class.scope_id);

    // Push a block so that we can add scoped instructions to it.
    context_.inst_block_stack().Push();
    AddNameScopeImportRefs(import_scope, new_scope);
    new_class.body_block_id = context_.inst_block_stack().Pop();

    if (import_class.base_id.is_valid()) {
      new_class.base_id = base_const_id.inst_id();
      // Add the base scope to extended scopes.
      auto base_inst_id = context_.types().GetInstId(
          context_.insts()
              .GetAs<SemIR::BaseDecl>(new_class.base_id)
              .base_type_id);
      const auto& base_class = context_.classes().Get(
          context_.insts().GetAs<SemIR::ClassType>(base_inst_id).class_id);
      new_scope.extended_scopes.push_back(base_class.scope_id);
    }
    CARBON_CHECK(new_scope.extended_scopes.size() ==
                 import_scope.extended_scopes.size());
  }

  auto TryResolveTypedInst(SemIR::ClassDecl inst,
                           SemIR::ConstantId class_const_id) -> ResolveResult {
    const auto& import_class = import_ir_.classes().Get(inst.class_id);

    // On the first pass, create a forward declaration of the class for any
    // recursive references.
    if (!class_const_id.is_valid()) {
      class_const_id = MakeIncompleteClass(import_class);
    }

    // Load constants for the definition.
    auto initial_work = work_stack_.size();

    auto enclosing_scope_id =
        GetLocalNameScopeId(import_class.enclosing_scope_id);
    auto object_repr_const_id =
        import_class.object_repr_id.is_valid()
            ? GetLocalConstantId(import_class.object_repr_id)
            : SemIR::ConstantId::Invalid;
    auto base_const_id = import_class.base_id.is_valid()
                             ? GetLocalConstantId(import_class.base_id)
                             : SemIR::ConstantId::Invalid;

    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry(class_const_id);
    }

    auto& new_class = context_.classes().Get(
        context_.insts()
            .GetAs<SemIR::ClassType>(class_const_id.inst_id())
            .class_id);
    new_class.enclosing_scope_id = enclosing_scope_id;

    if (import_class.is_defined()) {
      AddClassDefinition(import_class, new_class, object_repr_const_id,
                         base_const_id);
    }

    return {class_const_id};
  }

  auto TryResolveTypedInst(SemIR::ClassType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id =
        GetLocalConstantId(import_ir_.classes().Get(inst.class_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    return {class_const_id};
  }

  auto TryResolveTypedInst(SemIR::ConstType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto inner_const_id = GetLocalConstantId(inst.inner_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto inner_type_id = context_.GetTypeIdForTypeConstant(inner_const_id);
    // TODO: Should ConstType have a wrapper for this similar to the others?
    return {
        TryEvalInst(context_, SemIR::InstId::Invalid,
                    SemIR::ConstType{SemIR::TypeId::TypeType, inner_type_id})};
  }

  auto TryResolveTypedInst(SemIR::FieldDecl inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto inst_id = context_.AddInstInNoBlock(SemIR::LocIdAndInst::Untyped(
        AddImportIRInst(import_inst_id),
        SemIR::FieldDecl{context_.GetTypeIdForTypeConstant(const_id),
                         GetLocalNameId(inst.name_id), inst.index}));
    return {context_.constant_values().Get(inst_id)};
  }

  auto TryResolveTypedInst(SemIR::FunctionDecl inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);

    const auto& function = import_ir_.functions().Get(inst.function_id);
    auto return_type_const_id = SemIR::ConstantId::Invalid;
    if (function.return_type_id.is_valid()) {
      return_type_const_id = GetLocalConstantId(function.return_type_id);
    }
    auto enclosing_scope_id = GetLocalNameScopeId(function.enclosing_scope_id);
    llvm::SmallVector<SemIR::ConstantId> implicit_param_const_ids =
        GetLocalParamConstantIds(function.implicit_param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> param_const_ids =
        GetLocalParamConstantIds(function.param_refs_id);

    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Add the function declaration.
    auto function_decl = SemIR::FunctionDecl{
        context_.GetTypeIdForTypeConstant(type_const_id),
        SemIR::FunctionId::Invalid, SemIR::InstBlockId::Empty};
    // Prefer pointing diagnostics towards a definition.
    auto imported_loc_id = AddImportIRInst(function.definition_id.is_valid()
                                               ? function.definition_id
                                               : function.decl_id);
    auto function_decl_id = context_.AddPlaceholderInstInNoBlock(
        SemIR::LocIdAndInst::Untyped(imported_loc_id, function_decl));

    auto new_return_type_id =
        return_type_const_id.is_valid()
            ? context_.GetTypeIdForTypeConstant(return_type_const_id)
            : SemIR::TypeId::Invalid;
    auto new_return_storage = SemIR::InstId::Invalid;
    if (function.return_storage_id.is_valid()) {
      // Recreate the return slot from scratch.
      // TODO: Once we import function definitions, we'll need to make sure we
      // use the same return storage variable in the declaration and definition.
      new_return_storage = context_.AddInstInNoBlock(
          {AddImportIRInst(function.return_storage_id),
           SemIR::VarStorage{new_return_type_id, SemIR::NameId::ReturnSlot}});
    }
    function_decl.function_id = context_.functions().Add(
        {.name_id = GetLocalNameId(function.name_id),
         .enclosing_scope_id = enclosing_scope_id,
         .decl_id = function_decl_id,
         .implicit_param_refs_id = GetLocalParamRefsId(
             function.implicit_param_refs_id, implicit_param_const_ids),
         .param_refs_id =
             GetLocalParamRefsId(function.param_refs_id, param_const_ids),
         .return_type_id = new_return_type_id,
         .return_storage_id = new_return_storage,
         .is_extern = function.is_extern,
         .return_slot = function.return_slot,
         .builtin_kind = function.builtin_kind,
         .definition_id = function.definition_id.is_valid()
                              ? function_decl_id
                              : SemIR::InstId::Invalid});
    // Write the function ID into the FunctionDecl.
    context_.ReplaceInstBeforeConstantUse(function_decl_id, function_decl);
    return {context_.constant_values().Get(function_decl_id)};
  }

  auto TryResolveTypedInst(SemIR::ImportRefUsed /*inst*/, SemIR::InstId inst_id)
      -> ResolveResult {
    auto initial_work = work_stack_.size();
    // Return the constant for the instruction of the imported constant.
    auto constant_id = import_ir_.constant_values().Get(inst_id);
    if (!constant_id.is_valid()) {
      return {SemIR::ConstantId::Error};
    }
    if (!constant_id.is_constant()) {
      context_.TODO(inst_id, "Non-constant ImportRefUsed (comes up with var)");
      return {SemIR::ConstantId::Error};
    }

    auto new_constant_id = GetLocalConstantId(constant_id.inst_id());
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return {new_constant_id};
  }

  // Make a declaration of an interface. This is done as a separate step from
  // importing the interface definition in order to resolve cycles.
  auto MakeInterfaceDecl(const SemIR::Interface& import_interface)
      -> SemIR::ConstantId {
    auto interface_decl = SemIR::InterfaceDecl{SemIR::TypeId::Invalid,
                                               SemIR::InterfaceId::Invalid,
                                               SemIR::InstBlockId::Empty};
    auto interface_decl_id =
        context_.AddPlaceholderInst(SemIR::LocIdAndInst::Untyped(
            AddImportIRInst(import_interface.decl_id), interface_decl));

    // Start with an incomplete interface.
    SemIR::Interface new_interface = {
        .name_id = GetLocalNameId(import_interface.name_id),
        // Set in the second pass once we've imported it.
        .enclosing_scope_id = SemIR::NameScopeId::Invalid,
        .decl_id = interface_decl_id,
    };

    // Write the interface ID into the InterfaceDecl.
    interface_decl.interface_id = context_.interfaces().Add(new_interface);
    context_.ReplaceInstBeforeConstantUse(interface_decl_id, interface_decl);

    // Set the constant value for the imported interface.
    return context_.constant_values().Get(interface_decl_id);
  }

  // Imports the definition for an interface that has been imported as a forward
  // declaration.
  auto AddInterfaceDefinition(const SemIR::Interface& import_interface,
                              SemIR::Interface& new_interface,
                              SemIR::ConstantId self_param_id) -> void {
    new_interface.scope_id = context_.name_scopes().Add(
        new_interface.decl_id, SemIR::NameId::Invalid,
        new_interface.enclosing_scope_id);
    auto& new_scope = context_.name_scopes().Get(new_interface.scope_id);
    const auto& import_scope =
        import_ir_.name_scopes().Get(import_interface.scope_id);

    // Push a block so that we can add scoped instructions to it.
    context_.inst_block_stack().Push();
    AddNameScopeImportRefs(import_scope, new_scope);
    new_interface.associated_entities_id =
        AddAssociatedEntities(import_interface.associated_entities_id);
    new_interface.body_block_id = context_.inst_block_stack().Pop();
    new_interface.self_param_id = self_param_id.inst_id();

    CARBON_CHECK(import_scope.extended_scopes.empty())
        << "Interfaces don't currently have extended scopes to support.";
  }

  auto TryResolveTypedInst(SemIR::InterfaceDecl inst,
                           SemIR::ConstantId interface_const_id)
      -> ResolveResult {
    const auto& import_interface =
        import_ir_.interfaces().Get(inst.interface_id);

    // On the first pass, create a forward declaration of the interface.
    if (!interface_const_id.is_valid()) {
      interface_const_id = MakeInterfaceDecl(import_interface);
    }

    auto initial_work = work_stack_.size();

    auto enclosing_scope_id =
        GetLocalNameScopeId(import_interface.enclosing_scope_id);
    auto self_param_id = GetLocalConstantId(import_interface.self_param_id);

    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry(interface_const_id);
    }

    auto& new_interface = context_.interfaces().Get(
        context_.insts()
            .GetAs<SemIR::InterfaceType>(interface_const_id.inst_id())
            .interface_id);
    new_interface.enclosing_scope_id = enclosing_scope_id;

    if (import_interface.is_defined()) {
      AddInterfaceDefinition(import_interface, new_interface, self_param_id);
    }
    return {interface_const_id};
  }

  auto TryResolveTypedInst(SemIR::InterfaceType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto interface_const_id = GetLocalConstantId(
        import_ir_.interfaces().Get(inst.interface_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    return {interface_const_id};
  }

  auto TryResolveTypedInst(SemIR::InterfaceWitness inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    llvm::SmallVector<SemIR::InstId> elements;
    auto import_elements = import_ir_.inst_blocks().Get(inst.elements_id);
    elements.reserve(import_elements.size());
    for (auto import_elem_id : import_elements) {
      if (auto const_id = GetLocalConstantId(import_elem_id);
          const_id.is_valid()) {
        elements.push_back(const_id.inst_id());
      }
    }
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    CARBON_CHECK(elements.size() == import_elements.size())
        << "Failed to import an element without adding new work.";

    auto elements_id = context_.inst_blocks().Add(elements);
    return {TryEvalInst(
        context_, SemIR::InstId::Invalid,
        SemIR::InterfaceWitness{
            context_.GetBuiltinType(SemIR::BuiltinKind::WitnessType),
            elements_id})};
  }

  auto TryResolveTypedInst(SemIR::PointerType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto pointee_const_id = GetLocalConstantId(inst.pointee_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    auto pointee_type_id = context_.GetTypeIdForTypeConstant(pointee_const_id);
    return {context_.types().GetConstantId(
        context_.GetPointerType(pointee_type_id))};
  }

  auto TryResolveTypedInst(SemIR::StructType inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    // Collect all constants first, locating unresolved ones in a single pass.
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto orig_fields = import_ir_.inst_blocks().Get(inst.fields_id);
    llvm::SmallVector<SemIR::ConstantId> field_const_ids;
    field_const_ids.reserve(orig_fields.size());
    for (auto field_id : orig_fields) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      field_const_ids.push_back(GetLocalConstantId(field.field_type_id));
    }
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Prepare a vector of fields for GetStructType.
    // TODO: Should we have field constants so that we can deduplicate fields
    // without creating instructions here?
    llvm::SmallVector<SemIR::InstId> fields;
    fields.reserve(orig_fields.size());
    for (auto [field_id, field_const_id] :
         llvm::zip(orig_fields, field_const_ids)) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto name_id = GetLocalNameId(field.name_id);
      auto field_type_id = context_.GetTypeIdForTypeConstant(field_const_id);
      fields.push_back(context_.AddInstInNoBlock(
          {AddImportIRInst(import_inst_id),
           SemIR::StructTypeField{.name_id = name_id,
                                  .field_type_id = field_type_id}}));
    }

    return {context_.types().GetConstantId(
        context_.GetStructType(context_.inst_blocks().Add(fields)))};
  }

  auto TryResolveTypedInst(SemIR::TupleType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    // Collect all constants first, locating unresolved ones in a single pass.
    auto initial_work = work_stack_.size();
    auto orig_elem_type_ids = import_ir_.type_blocks().Get(inst.elements_id);
    llvm::SmallVector<SemIR::ConstantId> elem_const_ids;
    elem_const_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_type_id : orig_elem_type_ids) {
      elem_const_ids.push_back(GetLocalConstantId(elem_type_id));
    }
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Prepare a vector of the tuple types for GetTupleType.
    llvm::SmallVector<SemIR::TypeId> elem_type_ids;
    elem_type_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_const_id : elem_const_ids) {
      elem_type_ids.push_back(context_.GetTypeIdForTypeConstant(elem_const_id));
    }

    return {
        context_.types().GetConstantId(context_.GetTupleType(elem_type_ids))};
  }

  auto TryResolveTypedInst(SemIR::UnboundElementType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id = GetLocalConstantId(inst.class_type_id);
    auto elem_const_id = GetLocalConstantId(inst.element_type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return {context_.types().GetConstantId(context_.GetUnboundElementType(
        context_.GetTypeIdForTypeConstant(class_const_id),
        context_.GetTypeIdForTypeConstant(elem_const_id)))};
  }

  Context& context_;
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File& import_ir_;
  SemIR::ConstantValueStore& import_ir_constant_values_;
  llvm::SmallVector<Work> work_stack_;
};

// Returns the LocId corresponding to the input location.
static auto SemIRLocToLocId(Context& context, SemIRLoc loc) -> SemIR::LocId {
  if (loc.is_inst_id) {
    return context.insts().GetLocId(loc.inst_id);
  } else {
    return loc.loc_id;
  }
}

// Replace the ImportRefUnloaded instruction with an appropriate ImportRef. This
// doesn't use ReplaceInstBeforeConstantUse because it would trigger
// TryEvalInst, which we want to avoid with ImportRefs.
static auto SetInst(Context& context, SemIR::InstId inst_id,
                    SemIR::TypeId type_id,
                    SemIR::ImportIRInstId import_ir_inst_id, SemIRLoc loc,
                    bool set_if_invalid_loc) -> void {
  if (auto loc_id = SemIRLocToLocId(context, loc); loc_id.is_valid()) {
    context.sem_ir().insts().Set(
        inst_id, SemIR::ImportRefUsed{type_id, import_ir_inst_id, loc_id});
  } else if (set_if_invalid_loc) {
    context.sem_ir().insts().Set(
        inst_id, SemIR::ImportRefLoaded{type_id, import_ir_inst_id});
  }
}

auto LoadImportRef(Context& context, SemIR::InstId inst_id, SemIRLoc loc)
    -> void {
  CARBON_KIND_SWITCH(context.insts().Get(inst_id)) {
    case CARBON_KIND(SemIR::ImportRefLoaded inst): {
      SetInst(context, inst_id, inst.type_id, inst.import_ir_inst_id, loc,
              /*set_if_invalid_loc=*/false);
      return;
    }
    case CARBON_KIND(SemIR::ImportRefUnloaded inst): {
      auto import_ir_inst =
          context.import_ir_insts().Get(inst.import_ir_inst_id);

      const SemIR::File& import_ir =
          *context.import_irs().Get(import_ir_inst.ir_id).sem_ir;
      auto import_inst = import_ir.insts().Get(import_ir_inst.inst_id);

      ImportRefResolver resolver(context, import_ir_inst.ir_id);
      auto type_id = resolver.ResolveType(import_inst.type_id());
      auto constant_id = resolver.Resolve(import_ir_inst.inst_id);

      SetInst(context, inst_id, type_id, inst.import_ir_inst_id, loc,
              /*set_if_invalid_loc=*/true);

      // Store the constant for both the ImportRefUsed and imported instruction.
      context.constant_values().Set(inst_id, constant_id);
      return;
    }
    default:
      return;
  }
}

// Imports the impl `import_impl_id` from the imported IR `import_ir`.
static auto ImportImpl(Context& context, SemIR::ImportIRId import_ir_id,
                       const SemIR::File& import_ir,
                       SemIR::ImplId import_impl_id) -> void {
  // Resolve the imported impl to a local impl ID.
  ImportRefResolver resolver(context, import_ir_id);
  const auto& import_impl = import_ir.impls().Get(import_impl_id);
  auto self_id = resolver.ResolveType(import_impl.self_id);
  auto constraint_id = resolver.ResolveType(import_impl.constraint_id);

  // Import the definition if the impl is defined.
  // TODO: Do we need to check for multiple definitions?
  auto impl_id = context.impls().LookupOrAdd(self_id, constraint_id);
  if (import_impl.is_defined()) {
    // TODO: Create a scope for the `impl` if necessary.
    // TODO: Consider importing the definition_id.

    auto& impl = context.impls().Get(impl_id);
    impl.witness_id = context.AddImportRef(
        {.ir_id = import_ir_id, .inst_id = import_impl.witness_id});
  }
}

// TODO: This doesn't belong in this file. Consider moving the import resolver
// and this file elsewhere.
auto ImportImpls(Context& context) -> void {
  for (auto [import_index, import_ir] :
       llvm::enumerate(context.import_irs().array_ref())) {
    if (!import_ir.sem_ir) {
      continue;
    }

    auto import_ir_id = SemIR::ImportIRId(import_index);
    for (auto impl_index : llvm::seq(import_ir.sem_ir->impls().size())) {
      auto impl_id = SemIR::ImplId(impl_index);
      ImportImpl(context, import_ir_id, *import_ir.sem_ir, impl_id);
    }
  }
}

}  // namespace Carbon::Check
