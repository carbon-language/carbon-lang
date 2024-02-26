// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_ref.h"

#include "common/check.h"
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
// 1. If Work has received a constant, it's considered resolved.
//    - If made_incomplete_type, resolve unconditionally.
//    - The constant check avoids performance costs of deduplication on add.
// 2. Resolve the instruction: (TryResolveInst/TryResolveTypedInst)
//    - For most cases:
//      A. For types which _can_ be incomplete, when not made_incomplete_type:
//        i. Start by making an incomplete type to address circular references.
//        ii. If the imported type is incomplete, return the constant.
//        iii. Otherwise, set made_incomplete_type and continue resolving.
//          - Creating an incomplete type will have set the constant, which
//            influences step (1); setting made_incomplete_type gets us a second
//            resolve pass when needed.
//      B. Gather all input constants.
//        - Gathering constants directly adds unresolved values to work_stack_.
//      C. If any need to be resolved (HasNewWork), return Invalid; this
//         instruction needs two calls to complete.
//      D. Build any necessary IR structures, and return the output constant.
//    - For trivial cases with zero or one input constants, this may return
//      a constant (if one, potentially Invalid) directly.
// 3. If resolving returned a non-Invalid constant, pop the work; otherwise, it
//    needs to remain (and may no longer be at the top of the stack).
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
        import_ir_(*context_.import_irs().Get(import_ir_id)),
        import_ir_constant_values_(
            context_.import_ir_constant_values()[import_ir_id.index]) {}

  // Iteratively resolves an imported instruction's inner references until a
  // constant ID referencing the current IR is produced. When an outer
  // instruction has unresolved inner references, it will add them to the stack
  // for inner evaluation and reattempt outer evaluation after.
  auto Resolve(SemIR::InstId inst_id) -> SemIR::ConstantId {
    work_stack_.push_back({inst_id});
    while (!work_stack_.empty()) {
      auto work = work_stack_.back();
      CARBON_CHECK(work.inst_id.is_valid());

      // Double-check that the constant still doesn't have a calculated value.
      // This should typically be checked before adding it, but a given
      // instruction may be added multiple times before its constant is
      // evaluated.
      if (!work.made_incomplete_type &&
          import_ir_constant_values_.Get(work.inst_id).is_valid()) {
        work_stack_.pop_back();
      } else if (auto new_const_id =
                     TryResolveInst(work.inst_id, work.made_incomplete_type);
                 new_const_id.is_valid()) {
        import_ir_constant_values_.Set(work.inst_id, new_const_id);
        work_stack_.pop_back();
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

    // True if a first pass made an incomplete type.
    bool made_incomplete_type = false;
  };

  // For imported entities, we use an invalid enclosing scope. This will be okay
  // if the scope isn't used later, but we may need to change logic for this if
  // the behavior changes.
  static constexpr SemIR::NameScopeId NoEnclosingScopeForImports =
      SemIR::NameScopeId::Invalid;

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
      if (addr_inst) {
        inst = import_ir_.insts().Get(addr_inst->inner_id);
      }
      auto bind_inst = inst.TryAs<SemIR::AnyBindName>();
      if (bind_inst) {
        inst = import_ir_.insts().Get(bind_inst->value_id);
      }
      auto param_inst = inst.As<SemIR::Param>();

      // Rebuild the param instruction.
      auto name_id = GetLocalNameId(param_inst.name_id);
      auto type_id = context_.GetTypeIdForTypeConstant(const_id);

      auto new_param_id = context_.AddInstInNoBlock(
          {Parse::NodeId::Invalid, SemIR::Param{type_id, name_id}});
      if (bind_inst) {
        auto bind_name_id = context_.bind_names().Add(
            {.name_id = name_id,
             .enclosing_scope_id = SemIR::NameScopeId::Invalid});
        switch (bind_inst->kind) {
          case SemIR::InstKind::BindName:
            new_param_id = context_.AddInstInNoBlock(
                {Parse::NodeId::Invalid,
                 SemIR::BindName{type_id, bind_name_id, new_param_id}});
            break;
          case SemIR::InstKind::BindSymbolicName:
            new_param_id = context_.AddInstInNoBlock(
                {Parse::NodeId::Invalid,
                 SemIR::BindSymbolicName{type_id, bind_name_id, new_param_id}});
            break;

          default:
            CARBON_FATAL() << "Unexpected kind: " << bind_inst->kind;
        }
      }
      if (addr_inst) {
        new_param_id = context_.AddInstInNoBlock(
            {Parse::NodeId::Invalid,
             SemIR::AddrPattern{type_id, new_param_id}});
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

  // Tries to resolve the InstId, returning a constant when ready, or Invalid if
  // more has been added to the stack. A similar API is followed for all
  // following TryResolveTypedInst helper functions.
  //
  // TODO: Error is returned when support is missing, but that should go away.
  auto TryResolveInst(SemIR::InstId inst_id, bool made_incomplete_type)
      -> SemIR::ConstantId {
    if (inst_id.is_builtin()) {
      CARBON_CHECK(!made_incomplete_type);
      // Constants for builtins can be directly copied.
      return context_.constant_values().Get(inst_id);
    }

    auto inst = import_ir_.insts().Get(inst_id);

    CARBON_CHECK(!made_incomplete_type ||
                 inst.kind() == SemIR::InstKind::ClassDecl)
        << "Currently only decls with incomplete types should need "
           "made_incomplete_type states: "
        << inst.kind();

    switch (inst.kind()) {
      case SemIR::InstKind::BaseDecl:
        return TryResolveTypedInst(inst.As<SemIR::BaseDecl>());

      case SemIR::InstKind::BindAlias:
        return TryResolveTypedInst(inst.As<SemIR::BindAlias>());

      case SemIR::InstKind::ClassDecl:
        return TryResolveTypedInst(inst.As<SemIR::ClassDecl>(), inst_id,
                                   made_incomplete_type);

      case SemIR::InstKind::ClassType:
        return TryResolveTypedInst(inst.As<SemIR::ClassType>());

      case SemIR::InstKind::ConstType:
        return TryResolveTypedInst(inst.As<SemIR::ConstType>());

      case SemIR::InstKind::FieldDecl:
        return TryResolveTypedInst(inst.As<SemIR::FieldDecl>());

      case SemIR::InstKind::FunctionDecl:
        return TryResolveTypedInst(inst.As<SemIR::FunctionDecl>());

      case SemIR::InstKind::PointerType:
        return TryResolveTypedInst(inst.As<SemIR::PointerType>());

      case SemIR::InstKind::StructType:
        return TryResolveTypedInst(inst.As<SemIR::StructType>());

      case SemIR::InstKind::TupleType:
        return TryResolveTypedInst(inst.As<SemIR::TupleType>());

      case SemIR::InstKind::UnboundElementType:
        return TryResolveTypedInst(inst.As<SemIR::UnboundElementType>());

      case SemIR::InstKind::BindName:
      case SemIR::InstKind::BindSymbolicName:
        // Can use TryEvalInst because the resulting constant doesn't really use
        // `inst`.
        return TryEvalInst(context_, inst_id, inst);

      case SemIR::InstKind::InterfaceDecl:
        // TODO: Not implemented.
        return SemIR::ConstantId::Error;

      default:
        context_.TODO(
            Parse::NodeId::Invalid,
            llvm::formatv("TryResolveInst on {0}", inst.kind()).str());
        return SemIR::ConstantId::Error;
    }
  }

  auto TryResolveTypedInst(SemIR::BaseDecl inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);
    auto base_type_const_id = GetLocalConstantId(inst.base_type_id);
    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    // Import the instruction in order to update contained base_type_id.
    auto inst_id = context_.AddInstInNoBlock(
        {Parse::NodeId::Invalid,
         SemIR::BaseDecl{context_.GetTypeIdForTypeConstant(type_const_id),
                         context_.GetTypeIdForTypeConstant(base_type_const_id),
                         inst.index}});
    return context_.constant_values().Get(inst_id);
  }

  auto TryResolveTypedInst(SemIR::BindAlias inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    auto value_id = GetLocalConstantId(inst.value_id);
    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }
    return value_id;
  }

  // Makes an incomplete class. This is necessary even with classes with a
  // complete declaration, because things such as `Self` may refer back to the
  // type.
  auto MakeIncompleteClass(SemIR::InstId inst_id,
                           const SemIR::Class& import_class)
      -> SemIR::ConstantId {
    auto class_decl =
        SemIR::ClassDecl{SemIR::TypeId::Invalid, SemIR::ClassId::Invalid,
                         SemIR::InstBlockId::Empty};
    auto class_decl_id =
        context_.AddPlaceholderInst({Parse::NodeId::Invalid, class_decl});
    // Regardless of whether ClassDecl is a complete type, we first need an
    // incomplete type so that any references have something to point at.
    class_decl.class_id = context_.classes().Add({
        .name_id = GetLocalNameId(import_class.name_id),
        .enclosing_scope_id = NoEnclosingScopeForImports,
        // `.self_type_id` depends on the ClassType, so is set below.
        .self_type_id = SemIR::TypeId::Invalid,
        .decl_id = class_decl_id,
        .inheritance_kind = import_class.inheritance_kind,
    });

    // Write the function ID into the ClassDecl.
    context_.ReplaceInstBeforeConstantUse(class_decl_id,
                                          {Parse::NodeId::Invalid, class_decl});
    auto self_const_id = context_.constant_values().Get(class_decl_id);

    // Build the `Self` type using the resulting type constant.
    auto& class_info = context_.classes().Get(class_decl.class_id);
    class_info.self_type_id = context_.GetTypeIdForTypeConstant(self_const_id);

    // Set a constant corresponding to the incomplete class.
    import_ir_constant_values_.Set(inst_id, self_const_id);
    return self_const_id;
  }

  // Fills out the class definition for an incomplete class.
  auto AddClassDefinition(const SemIR::Class& import_class,
                          SemIR::ConstantId class_const_id,
                          SemIR::ConstantId object_repr_const_id,
                          SemIR::ConstantId base_const_id) -> void {
    auto& new_class = context_.classes().Get(
        context_.insts()
            .GetAs<SemIR::ClassType>(class_const_id.inst_id())
            .class_id);

    new_class.object_repr_id =
        context_.GetTypeIdForTypeConstant(object_repr_const_id);

    new_class.scope_id =
        context_.name_scopes().Add(new_class.decl_id, SemIR::NameId::Invalid,
                                   new_class.enclosing_scope_id);
    auto& new_scope = context_.name_scopes().Get(new_class.scope_id);
    const auto& old_scope = import_ir_.name_scopes().Get(import_class.scope_id);
    // Push a block so that we can add scoped instructions to it, primarily for
    // textual IR formatting.
    context_.inst_block_stack().Push();
    for (auto [entry_name_id, entry_inst_id] : old_scope.names) {
      CARBON_CHECK(
          new_scope.names
              .insert({GetLocalNameId(entry_name_id),
                       context_.AddPlaceholderInst(SemIR::ImportRefUnused{
                           import_ir_id_, entry_inst_id})})
              .second);
    }
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
                 old_scope.extended_scopes.size());
  }

  auto TryResolveTypedInst(SemIR::ClassDecl inst, SemIR::InstId inst_id,
                           bool made_incomplete_type) -> SemIR::ConstantId {
    const auto& import_class = import_ir_.classes().Get(inst.class_id);

    SemIR::ConstantId class_const_id = SemIR::ConstantId::Invalid;
    // On the first pass, there's no incomplete type; start by adding one for
    // any recursive references.
    if (!made_incomplete_type) {
      class_const_id = MakeIncompleteClass(inst_id, import_class);
      // If there's only a forward declaration, we're done.
      if (!import_class.object_repr_id.is_valid()) {
        return class_const_id;
      }
      // This may not be needed because all constants might be ready, but we do
      // it here so that we don't need to track which work item corresponds to
      // this instruction.
      work_stack_.back().made_incomplete_type = true;
    }

    CARBON_CHECK(import_class.object_repr_id.is_valid())
        << "Only reachable when there's a definition.";

    // Load constants for the definition.
    auto initial_work = work_stack_.size();

    auto object_repr_const_id = GetLocalConstantId(import_class.object_repr_id);
    auto base_const_id = import_class.base_id.is_valid()
                             ? GetLocalConstantId(import_class.base_id)
                             : SemIR::ConstantId::Invalid;

    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    // On the first pass, we build the incomplete type's constant above. If we
    // get here on a subsequent pass we need to fetch the one we built in the
    // first pass.
    if (made_incomplete_type) {
      CARBON_CHECK(!class_const_id.is_valid())
          << "Shouldn't have a const yet when resuming";
      class_const_id = import_ir_constant_values_.Get(inst_id);
    }
    AddClassDefinition(import_class, class_const_id, object_repr_const_id,
                       base_const_id);

    return class_const_id;
  }

  auto TryResolveTypedInst(SemIR::ClassType inst) -> SemIR::ConstantId {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    // ClassType uses a straight reference to the constant ID generated as part
    // of pulling in the ClassDecl, so there's no need to phase logic.
    return GetLocalConstantId(import_ir_.classes().Get(inst.class_id).decl_id);
  }

  auto TryResolveTypedInst(SemIR::ConstType inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto inner_const_id = GetLocalConstantId(inst.inner_id);
    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }
    auto inner_type_id = context_.GetTypeIdForTypeConstant(inner_const_id);
    // TODO: Should ConstType have a wrapper for this similar to the others?
    return TryEvalInst(
        context_, SemIR::InstId::Invalid,
        SemIR::ConstType{SemIR::TypeId::TypeType, inner_type_id});
  }

  auto TryResolveTypedInst(SemIR::FieldDecl inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    auto const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }
    auto inst_id = context_.AddInstInNoBlock(
        {Parse::NodeId::Invalid,
         SemIR::FieldDecl{context_.GetTypeIdForTypeConstant(const_id),
                          GetLocalNameId(inst.name_id), inst.index}});
    return context_.constant_values().Get(inst_id);
  }

  auto TryResolveTypedInst(SemIR::FunctionDecl inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);

    const auto& function = import_ir_.functions().Get(inst.function_id);
    auto return_type_const_id = SemIR::ConstantId::Invalid;
    if (function.return_type_id.is_valid()) {
      return_type_const_id = GetLocalConstantId(function.return_type_id);
    }
    auto return_slot_const_id = SemIR::ConstantId::Invalid;
    if (function.return_slot_id.is_valid()) {
      return_slot_const_id = GetLocalConstantId(function.return_slot_id);
    }
    llvm::SmallVector<SemIR::ConstantId> implicit_param_const_ids =
        GetLocalParamConstantIds(function.implicit_param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> param_const_ids =
        GetLocalParamConstantIds(function.param_refs_id);

    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    // Add the function declaration.
    auto function_decl =
        SemIR::FunctionDecl{context_.GetTypeIdForTypeConstant(type_const_id),
                            SemIR::FunctionId::Invalid};
    auto function_decl_id = context_.AddPlaceholderInstInNoBlock(
        {Parse::NodeId::Invalid, function_decl});

    auto new_return_type_id =
        return_type_const_id.is_valid()
            ? context_.GetTypeIdForTypeConstant(return_type_const_id)
            : SemIR::TypeId::Invalid;
    auto new_return_slot = SemIR::InstId::Invalid;
    if (function.return_slot_id.is_valid()) {
      context_.AddInstInNoBlock({SemIR::ImportRefUsed{
          context_.GetTypeIdForTypeConstant(return_slot_const_id),
          import_ir_id_, function.return_slot_id}});
    }
    function_decl.function_id = context_.functions().Add(
        {.name_id = GetLocalNameId(function.name_id),
         .enclosing_scope_id = NoEnclosingScopeForImports,
         .decl_id = function_decl_id,
         .implicit_param_refs_id = GetLocalParamRefsId(
             function.implicit_param_refs_id, implicit_param_const_ids),
         .param_refs_id =
             GetLocalParamRefsId(function.param_refs_id, param_const_ids),
         .return_type_id = new_return_type_id,
         .return_slot_id = new_return_slot});
    // Write the function ID into the FunctionDecl.
    context_.ReplaceInstBeforeConstantUse(
        function_decl_id, {Parse::NodeId::Invalid, function_decl});
    return context_.constant_values().Get(function_decl_id);
  }

  auto TryResolveTypedInst(SemIR::PointerType inst) -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto pointee_const_id = GetLocalConstantId(inst.pointee_id);
    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    auto pointee_type_id = context_.GetTypeIdForTypeConstant(pointee_const_id);
    return context_.types().GetConstantId(
        context_.GetPointerType(pointee_type_id));
  }

  auto TryResolveTypedInst(SemIR::StructType inst) -> SemIR::ConstantId {
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
      return SemIR::ConstantId::Invalid;
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
          {Parse::NodeId::Invalid,
           SemIR::StructTypeField{.name_id = name_id,
                                  .field_type_id = field_type_id}}));
    }

    return context_.types().GetConstantId(
        context_.GetStructType(context_.inst_blocks().Add(fields)));
  }

  auto TryResolveTypedInst(SemIR::TupleType inst) -> SemIR::ConstantId {
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
      return SemIR::ConstantId::Invalid;
    }

    // Prepare a vector of the tuple types for GetTupleType.
    llvm::SmallVector<SemIR::TypeId> elem_type_ids;
    elem_type_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_const_id : elem_const_ids) {
      elem_type_ids.push_back(context_.GetTypeIdForTypeConstant(elem_const_id));
    }

    return context_.types().GetConstantId(context_.GetTupleType(elem_type_ids));
  }

  auto TryResolveTypedInst(SemIR::UnboundElementType inst)
      -> SemIR::ConstantId {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id = GetLocalConstantId(inst.class_type_id);
    auto elem_const_id = GetLocalConstantId(inst.element_type_id);
    if (HasNewWork(initial_work)) {
      return SemIR::ConstantId::Invalid;
    }

    return context_.types().GetConstantId(context_.GetUnboundElementType(
        context_.GetTypeIdForTypeConstant(class_const_id),
        context_.GetTypeIdForTypeConstant(elem_const_id)));
  }

  Context& context_;
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File& import_ir_;
  SemIR::ConstantValueStore& import_ir_constant_values_;
  llvm::SmallVector<Work> work_stack_;
};

auto TryResolveImportRefUnused(Context& context, SemIR::InstId inst_id)
    -> void {
  auto inst = context.insts().Get(inst_id);
  auto import_ref = inst.TryAs<SemIR::ImportRefUnused>();
  if (!import_ref) {
    return;
  }

  const SemIR::File& import_ir = *context.import_irs().Get(import_ref->ir_id);
  auto import_inst = import_ir.insts().Get(import_ref->inst_id);

  ImportRefResolver resolver(context, import_ref->ir_id);
  auto type_id = resolver.ResolveType(import_inst.type_id());
  auto constant_id = resolver.Resolve(import_ref->inst_id);

  // TODO: Once ClassDecl/InterfaceDecl are supported (no longer return Error),
  // remove this.
  if (constant_id == SemIR::ConstantId::Error) {
    type_id = SemIR::TypeId::Error;
  }

  // Replace the ImportRefUnused instruction with an ImportRefUsed. This doesn't
  // use ReplaceInstBeforeConstantUse because it would trigger TryEvalInst, and
  // we're instead doing constant evaluation here in order to minimize recursion
  // risks.
  context.sem_ir().insts().Set(
      inst_id,
      SemIR::ImportRefUsed{type_id, import_ref->ir_id, import_ref->inst_id});

  // Store the constant for both the ImportRefUsed and imported instruction.
  context.constant_values().Set(inst_id, constant_id);
}

}  // namespace Carbon::Check
