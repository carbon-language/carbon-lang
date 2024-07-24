// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_ref.h"

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Adds the ImportIR, excluding the update to the check_ir_map.
static auto InternalAddImportIR(Context& context, SemIR::ImportIR import_ir)
    -> SemIR::ImportIRId {
  context.import_ir_constant_values().push_back(
      SemIR::ConstantValueStore(SemIR::ConstantId::Invalid));
  return context.import_irs().Add(import_ir);
}

auto SetApiImportIR(Context& context, SemIR::ImportIR import_ir) -> void {
  auto ir_id = SemIR::ImportIRId::Invalid;
  if (import_ir.sem_ir != nullptr) {
    ir_id = AddImportIR(context, import_ir);
  } else {
    // We don't have a check_ir_id, so add without touching check_ir_map.
    ir_id = InternalAddImportIR(context, import_ir);
  }
  CARBON_CHECK(ir_id == SemIR::ImportIRId::ApiForImpl)
      << "ApiForImpl must be the first IR";
}

auto AddImportIR(Context& context, SemIR::ImportIR import_ir)
    -> SemIR::ImportIRId {
  auto& ir_id = context.GetImportIRId(*import_ir.sem_ir);
  if (!ir_id.is_valid()) {
    // Note this updates check_ir_map.
    ir_id = InternalAddImportIR(context, import_ir);
  } else if (import_ir.is_export) {
    // We're processing an `export import`. In case the IR was indirectly added
    // as a non-export, mark it as an export.
    context.import_irs().Get(ir_id).is_export = true;
  }
  return ir_id;
}

auto AddImportRef(Context& context, SemIR::ImportIRInst import_ir_inst,
                  SemIR::EntityNameId entity_name_id) -> SemIR::InstId {
  auto import_ir_inst_id = context.import_ir_insts().Add(import_ir_inst);
  SemIR::ImportRefUnloaded inst = {.import_ir_inst_id = import_ir_inst_id,
                                   .entity_name_id = entity_name_id};
  auto import_ref_id = context.AddPlaceholderInstInNoBlock(
      SemIR::LocIdAndInst(import_ir_inst_id, inst));

  // ImportRefs have a dedicated block because this may be called during
  // processing where the instruction shouldn't be inserted in the current inst
  // block.
  context.import_ref_ids().push_back(import_ref_id);
  return import_ref_id;
}

auto GetCanonicalImportIRInst(Context& context, const SemIR::File* cursor_ir,
                              SemIR::InstId cursor_inst_id)
    -> SemIR::ImportIRInst {
  while (true) {
    auto inst = cursor_ir->insts().Get(cursor_inst_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(SemIR::ExportDecl bind_export): {
        cursor_inst_id = bind_export.value_id;
        continue;
      }
      case SemIR::ImportRefLoaded::Kind:
      case SemIR::ImportRefUnloaded::Kind: {
        auto import_ref = inst.As<SemIR::AnyImportRef>();
        auto import_ir_inst =
            cursor_ir->import_ir_insts().Get(import_ref.import_ir_inst_id);
        cursor_ir = cursor_ir->import_irs().Get(import_ir_inst.ir_id).sem_ir;
        cursor_inst_id = import_ir_inst.inst_id;
        continue;
      }
      default: {
        auto ir_id = SemIR::ImportIRId::Invalid;
        if (cursor_ir != &context.sem_ir()) {
          // This uses AddImportIR in case it was indirectly found, which can
          // happen with two or more steps of exports.
          ir_id = AddImportIR(context, {.decl_id = SemIR::InstId::Invalid,
                                        .is_export = false,
                                        .sem_ir = cursor_ir});
        }
        return {.ir_id = ir_id, .inst_id = cursor_inst_id};
      }
    }
  }
}

auto VerifySameCanonicalImportIRInst(Context& context, SemIR::InstId prev_id,
                                     SemIR::ImportIRInst prev_import_ir_inst,
                                     SemIR::ImportIRId new_ir_id,
                                     const SemIR::File* new_import_ir,
                                     SemIR::InstId new_inst_id) -> void {
  auto new_import_ir_inst =
      GetCanonicalImportIRInst(context, new_import_ir, new_inst_id);
  if (new_import_ir_inst == prev_import_ir_inst) {
    return;
  }
  auto conflict_id =
      AddImportRef(context, {.ir_id = new_ir_id, .inst_id = new_inst_id},
                   SemIR::EntityNameId::Invalid);
  context.DiagnoseDuplicateName(conflict_id, prev_id);
}

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
//
// TODO: Fix class `extern` handling and merging, rewrite tests.
// - check/testdata/class/cross_package_import.carbon
// - check/testdata/class/extern.carbon
// TODO: Fix function `extern` handling and merging, rewrite tests.
// - check/testdata/function/declaration/import.carbon
// - check/testdata/packages/cross_package_import.carbon
class ImportRefResolver {
 public:
  explicit ImportRefResolver(Context& context, SemIR::ImportIRId import_ir_id)
      : context_(context),
        import_ir_id_(import_ir_id),
        import_ir_(*context_.import_irs().Get(import_ir_id).sem_ir) {}

  // Iteratively resolves an imported instruction's inner references until a
  // constant ID referencing the current IR is produced. See the class comment
  // for more details.
  auto Resolve(SemIR::InstId inst_id) -> SemIR::ConstantId {
    work_stack_.push_back({.inst_id = inst_id});
    while (!work_stack_.empty()) {
      auto work = work_stack_.back();
      CARBON_CHECK(work.inst_id.is_valid());

      // Step 1: check for a constant value.
      auto existing = FindResolvedConstId(work.inst_id);
      if (existing.const_id.is_valid() && !work.retry) {
        work_stack_.pop_back();
        continue;
      }

      // Step 2: resolve the instruction.
      auto initial_work = work_stack_.size();
      auto [new_const_id, finished] =
          TryResolveInst(work.inst_id, existing.const_id);
      CARBON_CHECK(finished == !HasNewWork(initial_work));

      CARBON_CHECK(!existing.const_id.is_valid() ||
                   existing.const_id == new_const_id)
          << "Constant value changed in second pass.";
      if (!existing.const_id.is_valid()) {
        SetResolvedConstId(work.inst_id, existing.indirect_insts, new_const_id);
      }

      // Step 3: pop or retry.
      if (finished) {
        work_stack_.pop_back();
      } else {
        work_stack_[initial_work - 1].retry = true;
      }
    }
    auto constant_id = import_ir_constant_values().Get(inst_id);
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
      return context_.GetBuiltinType(import_type_inst_id.builtin_inst_kind());
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

  // The constant found by FindResolvedConstId.
  struct ResolvedConstId {
    // The constant for the instruction. Invalid if not yet resolved.
    SemIR::ConstantId const_id = SemIR::ConstantId::Invalid;

    // Instructions which are indirect but equivalent to the current instruction
    // being resolved, and should have their constant set to the same. Empty
    // when const_id is valid.
    llvm::SmallVector<SemIR::ImportIRInst> indirect_insts = {};
  };

  // Looks to see if an instruction has been resolved. If a constant is only
  // found indirectly, sets the constant for any indirect steps that don't
  // already have the constant. If a constant isn't found, returns the indirect
  // instructions so that they can have the resolved constant assigned later.
  auto FindResolvedConstId(SemIR::InstId inst_id) -> ResolvedConstId {
    ResolvedConstId result;

    if (auto existing_const_id = import_ir_constant_values().Get(inst_id);
        existing_const_id.is_valid()) {
      result.const_id = existing_const_id;
      return result;
    }

    const auto* cursor_ir = &import_ir_;
    auto cursor_ir_id = SemIR::ImportIRId::Invalid;
    auto cursor_inst_id = inst_id;

    while (true) {
      auto loc_id = cursor_ir->insts().GetLocId(cursor_inst_id);
      if (!loc_id.is_import_ir_inst_id()) {
        return result;
      }
      auto ir_inst =
          cursor_ir->import_ir_insts().Get(loc_id.import_ir_inst_id());

      const auto* prev_ir = cursor_ir;
      auto prev_inst_id = cursor_inst_id;

      cursor_ir = cursor_ir->import_irs().Get(ir_inst.ir_id).sem_ir;
      cursor_ir_id = context_.GetImportIRId(*cursor_ir);
      if (!cursor_ir_id.is_valid()) {
        // TODO: Should we figure out a location to assign here?
        cursor_ir_id = AddImportIR(context_, {.decl_id = SemIR::InstId::Invalid,
                                              .is_export = false,
                                              .sem_ir = cursor_ir});
      }
      cursor_inst_id = ir_inst.inst_id;

      CARBON_CHECK(cursor_ir != prev_ir || cursor_inst_id != prev_inst_id)
          << cursor_ir->insts().Get(cursor_inst_id);

      if (auto const_id =
              context_.import_ir_constant_values()[cursor_ir_id.index].Get(
                  cursor_inst_id);
          const_id.is_valid()) {
        SetResolvedConstId(inst_id, result.indirect_insts, const_id);
        result.const_id = const_id;
        result.indirect_insts.clear();
        return result;
      } else {
        result.indirect_insts.push_back(
            {.ir_id = cursor_ir_id, .inst_id = cursor_inst_id});
      }
    }
  }

  // Sets a resolved constant into the current and indirect instructions.
  auto SetResolvedConstId(SemIR::InstId inst_id,
                          llvm::ArrayRef<SemIR::ImportIRInst> indirect_insts,
                          SemIR::ConstantId const_id) -> void {
    import_ir_constant_values().Set(inst_id, const_id);
    for (auto indirect_inst : indirect_insts) {
      context_.import_ir_constant_values()[indirect_inst.ir_id.index].Set(
          indirect_inst.inst_id, const_id);
    }
  }

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

  auto AddImportIRInst(SemIR::InstId inst_id) -> SemIR::ImportIRInstId {
    return context_.import_ir_insts().Add(
        {.ir_id = import_ir_id_, .inst_id = inst_id});
  }

  // Returns the ConstantId for an InstId. Adds unresolved constants to
  // work_stack_.
  auto GetLocalConstantId(SemIR::InstId inst_id) -> SemIR::ConstantId {
    auto const_id = import_ir_constant_values().Get(inst_id);
    if (!const_id.is_valid()) {
      work_stack_.push_back({.inst_id = inst_id});
    }
    return const_id;
  }

  // Returns the local constant InstId for an imported InstId.
  auto GetLocalConstantInstId(SemIR::InstId inst_id) -> SemIR::InstId {
    auto const_id = import_ir_constant_values().Get(inst_id);
    if (!const_id.is_valid()) {
      work_stack_.push_back({.inst_id = inst_id});
      return SemIR::InstId::Invalid;
    }
    return context_.constant_values().GetInstId(const_id);
  }

  // Returns the ConstantId for a TypeId. Adds unresolved constants to
  // work_stack_.
  auto GetLocalConstantId(SemIR::TypeId type_id) -> SemIR::ConstantId {
    return GetLocalConstantId(import_ir_.types().GetInstId(type_id));
  }

  // Gets the local constant values corresponding to an imported inst block.
  auto GetLocalInstBlockContents(SemIR::InstBlockId import_block_id)
      -> llvm::SmallVector<SemIR::InstId> {
    llvm::SmallVector<SemIR::InstId> inst_ids;
    if (!import_block_id.is_valid() ||
        import_block_id == SemIR::InstBlockId::Empty) {
      return inst_ids;
    }

    // Import all the values in the block.
    auto import_block = import_ir_.inst_blocks().Get(import_block_id);
    inst_ids.reserve(import_block.size());
    for (auto import_inst_id : import_block) {
      auto const_id = GetLocalConstantId(import_inst_id);
      inst_ids.push_back(context_.constant_values().GetInstIdIfValid(const_id));
    }

    return inst_ids;
  }

  // Gets a local canonical instruction block ID corresponding to an imported
  // inst block whose contents were already imported, for example by
  // GetLocalInstBlockContents.
  auto GetLocalCanonicalInstBlockId(SemIR::InstBlockId import_block_id,
                                    llvm::ArrayRef<SemIR::InstId> contents)
      -> SemIR::InstBlockId {
    if (!import_block_id.is_valid()) {
      return SemIR::InstBlockId::Invalid;
    }
    return context_.inst_blocks().AddCanonical(contents);
  }

  // Gets a local version of an imported generic.
  auto GetLocalGeneric(SemIR::GenericId generic_id) -> SemIR::GenericId {
    if (!generic_id.is_valid()) {
      return SemIR::GenericId::Invalid;
    }

    // TODO: Support importing generics. Note that this comes up in the prelude,
    // so for now we fall back to producing `Invalid` and treating imported
    // generics as non-generic.
    return SemIR::GenericId::Invalid;
  }

  // Gets a local argument list corresponding to the arguments of an imported
  // generic instance.
  auto GetLocalGenericInstanceArgs(SemIR::GenericInstanceId instance_id)
      -> llvm::SmallVector<SemIR::InstId> {
    if (!instance_id.is_valid()) {
      return {};
    }
    return GetLocalInstBlockContents(
        import_ir_.generic_instances().Get(instance_id).args_id);
  }

  // Gets a local generic instance whose arguments were already imported by
  // GetLocalGenericInstanceArgs. Does not add any new work.
  auto GetLocalGenericInstance(SemIR::GenericInstanceId instance_id,
                               llvm::ArrayRef<SemIR::InstId> args)
      -> SemIR::GenericInstanceId {
    if (!instance_id.is_valid()) {
      return SemIR::GenericInstanceId::Invalid;
    }
    const auto& instance = import_ir_.generic_instances().Get(instance_id);
    // TODO: Import the generic.
    auto generic_id = SemIR::GenericId::Invalid;
    auto args_id = GetLocalCanonicalInstBlockId(instance.args_id, args);
    // TODO: Also import the generic instance.
    return context_.generic_instances().GetOrAdd(generic_id, args_id);
  }

  // Returns the ConstantId for each parameter's type. Adds unresolved constants
  // to work_stack_.
  auto GetLocalParamConstantIds(SemIR::InstBlockId param_refs_id)
      -> llvm::SmallVector<SemIR::ConstantId> {
    llvm::SmallVector<SemIR::ConstantId> const_ids;
    if (!param_refs_id.is_valid() ||
        param_refs_id == SemIR::InstBlockId::Empty) {
      return const_ids;
    }

    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
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
    if (!param_refs_id.is_valid() ||
        param_refs_id == SemIR::InstBlockId::Empty) {
      return param_refs_id;
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

      auto new_param_id = context_.AddInstInNoBlock<SemIR::Param>(
          AddImportIRInst(param_id), {.type_id = type_id, .name_id = name_id});
      if (bind_inst) {
        switch (bind_inst->kind) {
          case SemIR::BindName::Kind: {
            auto entity_name_id = context_.entity_names().Add(
                {.name_id = name_id,
                 .parent_scope_id = SemIR::NameScopeId::Invalid,
                 .bind_index = SemIR::CompileTimeBindIndex::Invalid});
            new_param_id = context_.AddInstInNoBlock<SemIR::BindName>(
                AddImportIRInst(bind_id), {.type_id = type_id,
                                           .entity_name_id = entity_name_id,
                                           .value_id = new_param_id});
            break;
          }
          case SemIR::BindSymbolicName::Kind: {
            // The symbolic name will be created on first reference, so might
            // already exist. Update the value in it to refer to the parameter.
            auto new_bind_inst_id = GetLocalConstantInstId(bind_id);
            auto new_bind_inst =
                context_.insts().GetAs<SemIR::BindSymbolicName>(
                    new_bind_inst_id);
            new_bind_inst.value_id = new_param_id;
            // This is not before constant use, but doesn't change the
            // constant value of the instruction.
            context_.ReplaceInstBeforeConstantUse(new_bind_inst_id,
                                                  new_bind_inst);
            new_param_id = new_bind_inst_id;
            break;
          }
          default: {
            CARBON_FATAL() << "Unexpected kind: " << bind_inst->kind;
          }
        }
      }
      if (addr_inst) {
        new_param_id = context_.AddInstInNoBlock<SemIR::AddrPattern>(
            AddImportIRInst(ref_id),
            {.type_id = type_id, .inner_id = new_param_id});
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
    // Get the instruction that created the scope.
    auto [inst_id, inst] =
        import_ir_.name_scopes().GetInstIfValid(name_scope_id);
    if (!inst) {
      // Map scopes that aren't associated with an instruction to invalid
      // scopes. For now, such scopes aren't used, and we don't have a good way
      // to remap them.
      return SemIR::NameScopeId::Invalid;
    }

    // Get the constant value for the scope.
    auto const_id = SemIR::ConstantId::Invalid;
    CARBON_KIND_SWITCH(*inst) {
      case SemIR::ImplDecl::Kind:
        // TODO: Import the scope for an `impl` definition.
        return SemIR::NameScopeId::Invalid;

      case SemIR::Namespace::Kind:
        // If the namespace has already been imported, we can use its constant.
        // However, if it hasn't, we use Invalid instead of adding it to the
        // work stack. That's expected to be okay when resolving references.
        const_id = import_ir_constant_values().Get(inst_id);
        break;

      default:
        const_id = GetLocalConstantId(inst_id);
    }
    if (!const_id.is_valid()) {
      return SemIR::NameScopeId::Invalid;
    }

    auto const_inst_id = context_.constant_values().GetInstId(const_id);
    auto name_scope_inst = context_.insts().Get(const_inst_id);
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
      case SemIR::StructValue::Kind: {
        auto type_inst = context_.types().GetAsInst(name_scope_inst.type_id());
        CARBON_KIND_SWITCH(type_inst) {
          case CARBON_KIND(SemIR::GenericClassType inst): {
            return context_.classes().Get(inst.class_id).scope_id;
          }
          case CARBON_KIND(SemIR::GenericInterfaceType inst): {
            return context_.interfaces().Get(inst.interface_id).scope_id;
          }
          default: {
            break;
          }
        }
        break;
      }
      default: {
        if (const_inst_id == SemIR::InstId::BuiltinError) {
          return SemIR::NameScopeId::Invalid;
        }
        break;
      }
    }
    CARBON_FATAL() << "Unexpected instruction kind for name scope: "
                   << name_scope_inst;
  }

  // Adds ImportRefUnloaded entries for members of the imported scope, for name
  // lookup.
  auto AddNameScopeImportRefs(const SemIR::NameScope& import_scope,
                              SemIR::NameScope& new_scope) -> void {
    for (auto entry : import_scope.names) {
      auto ref_id = AddImportRef(
          context_, {.ir_id = import_ir_id_, .inst_id = entry.inst_id},
          SemIR::EntityNameId::Invalid);
      new_scope.AddRequired({.name_id = GetLocalNameId(entry.name_id),
                             .inst_id = ref_id,
                             .access_kind = entry.access_kind});
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
          AddImportRef(context_, {.ir_id = import_ir_id_, .inst_id = inst_id},
                       SemIR::EntityNameId::Invalid));
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
      return {.const_id = context_.constant_values().Get(inst_id)};
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
        return {.const_id = TryEvalInst(context_, inst_id, inst)};
      }
      case CARBON_KIND(SemIR::BindSymbolicName inst): {
        return TryResolveTypedInst(inst);
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
      case CARBON_KIND(SemIR::ExportDecl inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::FieldDecl inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::FunctionDecl inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::FunctionType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::GenericClassType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::GenericInterfaceType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::ImportRefLoaded inst): {
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
      case CARBON_KIND(SemIR::IntLiteral inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::Namespace inst): {
        CARBON_FATAL() << "Namespaces shouldn't need resolution this way: "
                       << inst;
      }
      case CARBON_KIND(SemIR::PointerType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::StructType inst): {
        return TryResolveTypedInst(inst, inst_id);
      }
      case CARBON_KIND(SemIR::StructValue inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::TupleType inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::TupleValue inst): {
        return TryResolveTypedInst(inst);
      }
      case CARBON_KIND(SemIR::UnboundElementType inst): {
        return TryResolveTypedInst(inst);
      }
      default:
        context_.TODO(
            SemIR::LocId(AddImportIRInst(inst_id)),
            llvm::formatv("TryResolveInst on {0}", untyped_inst.kind()).str());
        return {.const_id = SemIR::ConstantId::Error};
    }
  }

  // Produce a resolve result for the given instruction that describes a
  // constant value. This should only be used for instructions that describe
  // constants, and not for instructions that represent declarations. For a
  // declaration, we need an associated location, so AddInstInNoBlock should be
  // used instead.
  auto ResolveAsUntyped(SemIR::Inst inst) -> ResolveResult {
    auto result = TryEvalInst(context_, SemIR::InstId::Invalid, inst);
    CARBON_CHECK(result.is_constant()) << inst << " is not constant";
    return {.const_id = result};
  }

  // Same as ResolveAsUntyped, but with an explicit type for convenience.
  template <typename InstT>
  auto ResolveAs(InstT inst) -> ResolveResult {
    return ResolveAsUntyped(inst);
  }

  auto TryResolveTypedInst(SemIR::AssociatedEntity inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Add a lazy reference to the target declaration.
    auto decl_id = AddImportRef(
        context_, {.ir_id = import_ir_id_, .inst_id = inst.decl_id},
        SemIR::EntityNameId::Invalid);

    return ResolveAs<SemIR::AssociatedEntity>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_const_id),
         .index = inst.index,
         .decl_id = decl_id});
  }

  auto TryResolveTypedInst(SemIR::AssociatedEntityType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    auto initial_work = work_stack_.size();
    auto entity_type_const_id = GetLocalConstantId(inst.entity_type_id);
    auto interface_inst_id = GetLocalConstantInstId(
        import_ir_.interfaces().Get(inst.interface_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return ResolveAs<SemIR::AssociatedEntityType>(
        {.type_id = SemIR::TypeId::TypeType,
         .interface_id = context_.insts()
                             .GetAs<SemIR::InterfaceType>(interface_inst_id)
                             .interface_id,
         .entity_type_id =
             context_.GetTypeIdForTypeConstant(entity_type_const_id)});
  }

  auto TryResolveTypedInst(SemIR::BaseDecl inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_const_id = GetLocalConstantId(inst.type_id);
    auto base_type_const_id = GetLocalConstantId(inst.base_type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Import the instruction in order to update contained base_type_id and
    // track the import location.
    auto inst_id = context_.AddInstInNoBlock<SemIR::BaseDecl>(
        AddImportIRInst(import_inst_id),
        {.type_id = context_.GetTypeIdForTypeConstant(type_const_id),
         .base_type_id = context_.GetTypeIdForTypeConstant(base_type_const_id),
         .index = inst.index});
    return {.const_id = context_.constant_values().Get(inst_id)};
  }

  auto TryResolveTypedInst(SemIR::BindAlias inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto value_id = GetLocalConstantId(inst.value_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    return {.const_id = value_id};
  }

  auto TryResolveTypedInst(SemIR::BindSymbolicName inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    const auto& import_entity_name =
        import_ir_.entity_names().Get(inst.entity_name_id);
    auto name_id = GetLocalNameId(import_entity_name.name_id);
    auto entity_name_id = context_.entity_names().Add(
        {.name_id = name_id,
         .parent_scope_id = SemIR::NameScopeId::Invalid,
         .bind_index = import_entity_name.bind_index});
    return ResolveAs<SemIR::BindSymbolicName>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .entity_name_id = entity_name_id,
         .value_id = SemIR::InstId::Invalid});
  }

  // Makes an incomplete class. This is necessary even with classes with a
  // complete declaration, because things such as `Self` may refer back to the
  // type.
  auto MakeIncompleteClass(const SemIR::Class& import_class)
      -> std::pair<SemIR::ClassId, SemIR::ConstantId> {
    SemIR::ClassDecl class_decl = {.type_id = SemIR::TypeId::TypeType,
                                   .class_id = SemIR::ClassId::Invalid,
                                   .decl_block_id = SemIR::InstBlockId::Empty};
    auto class_decl_id = context_.AddPlaceholderInstInNoBlock(
        SemIR::LocIdAndInst(AddImportIRInst(import_class.decl_id), class_decl));
    // TODO: Support for importing generics.
    auto generic_id = GetLocalGeneric(import_class.generic_id);
    // Regardless of whether ClassDecl is a complete type, we first need an
    // incomplete type so that any references have something to point at.
    class_decl.class_id = context_.classes().Add({
        .name_id = GetLocalNameId(import_class.name_id),
        // These are set in the second pass once we've imported them. Import
        // enough of the parameter lists that we know whether this class is a
        // generic class and can build the right constant value for it.
        // TODO: Add a better way to represent a generic `Class` prior to
        // importing the parameters.
        .parent_scope_id = SemIR::NameScopeId::Invalid,
        .generic_id = generic_id,
        .first_param_node_id = Parse::NodeId::Invalid,
        .last_param_node_id = Parse::NodeId::Invalid,
        .implicit_param_refs_id = import_class.implicit_param_refs_id.is_valid()
                                      ? SemIR::InstBlockId::Empty
                                      : SemIR::InstBlockId::Invalid,
        .param_refs_id = import_class.param_refs_id.is_valid()
                             ? SemIR::InstBlockId::Empty
                             : SemIR::InstBlockId::Invalid,
        .self_type_id = SemIR::TypeId::Invalid,
        // These fields can be set immediately.
        .decl_id = class_decl_id,
        .inheritance_kind = import_class.inheritance_kind,
    });

    if (import_class.is_generic()) {
      class_decl.type_id = context_.GetGenericClassType(class_decl.class_id);
    }

    // Write the class ID into the ClassDecl.
    context_.ReplaceInstBeforeConstantUse(class_decl_id, class_decl);
    auto self_const_id = context_.constant_values().Get(class_decl_id);
    return {class_decl.class_id, self_const_id};
  }

  // Fills out the class definition for an incomplete class.
  auto AddClassDefinition(const SemIR::Class& import_class,
                          SemIR::Class& new_class,
                          SemIR::ConstantId object_repr_const_id,
                          SemIR::InstId base_id) -> void {
    new_class.definition_id = new_class.decl_id;

    new_class.object_repr_id =
        context_.GetTypeIdForTypeConstant(object_repr_const_id);

    new_class.scope_id = context_.name_scopes().Add(
        new_class.decl_id, SemIR::NameId::Invalid, new_class.parent_scope_id);
    auto& new_scope = context_.name_scopes().Get(new_class.scope_id);
    const auto& import_scope =
        import_ir_.name_scopes().Get(import_class.scope_id);

    // Push a block so that we can add scoped instructions to it.
    context_.inst_block_stack().Push();
    AddNameScopeImportRefs(import_scope, new_scope);
    new_class.body_block_id = context_.inst_block_stack().Pop();

    if (import_class.base_id.is_valid()) {
      new_class.base_id = base_id;
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
    // TODO: The handling of interfaces repeats a lot with the handling of
    // classes, and will likely also be repeated for named constraints and
    // choice types. Factor out some of this functionality.
    const auto& import_class = import_ir_.classes().Get(inst.class_id);

    SemIR::ClassId class_id = SemIR::ClassId::Invalid;
    if (!class_const_id.is_valid()) {
      // On the first pass, create a forward declaration of the class for any
      // recursive references.
      std::tie(class_id, class_const_id) = MakeIncompleteClass(import_class);
    } else {
      // On the second pass, compute the class ID from the constant value of the
      // declaration.
      auto class_const_inst = context_.insts().Get(
          context_.constant_values().GetInstId(class_const_id));
      if (auto class_type = class_const_inst.TryAs<SemIR::ClassType>()) {
        class_id = class_type->class_id;
      } else {
        auto generic_class_type =
            context_.types().GetAs<SemIR::GenericClassType>(
                class_const_inst.type_id());
        class_id = generic_class_type.class_id;
      }
    }

    // Load constants for the definition.
    auto initial_work = work_stack_.size();

    auto parent_scope_id = GetLocalNameScopeId(import_class.parent_scope_id);
    llvm::SmallVector<SemIR::ConstantId> implicit_param_const_ids =
        GetLocalParamConstantIds(import_class.implicit_param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> param_const_ids =
        GetLocalParamConstantIds(import_class.param_refs_id);
    auto self_const_id = GetLocalConstantId(import_class.self_type_id);
    auto object_repr_const_id =
        import_class.object_repr_id.is_valid()
            ? GetLocalConstantId(import_class.object_repr_id)
            : SemIR::ConstantId::Invalid;
    auto base_id = import_class.base_id.is_valid()
                       ? GetLocalConstantInstId(import_class.base_id)
                       : SemIR::InstId::Invalid;

    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry(class_const_id);
    }

    auto& new_class = context_.classes().Get(class_id);
    new_class.parent_scope_id = parent_scope_id;
    new_class.implicit_param_refs_id = GetLocalParamRefsId(
        import_class.implicit_param_refs_id, implicit_param_const_ids);
    new_class.param_refs_id =
        GetLocalParamRefsId(import_class.param_refs_id, param_const_ids);
    new_class.self_type_id = context_.GetTypeIdForTypeConstant(self_const_id);

    if (import_class.is_defined()) {
      AddClassDefinition(import_class, new_class, object_repr_const_id,
                         base_id);
    }

    return {.const_id = class_const_id};
  }

  auto TryResolveTypedInst(SemIR::ClassType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id =
        GetLocalConstantId(import_ir_.classes().Get(inst.class_id).decl_id);
    auto args = GetLocalGenericInstanceArgs(inst.instance_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Find the corresponding class type. For a non-generic class, this is the
    // type of the class declaration. For a generic class, build a class type
    // referencing this specialization of the generic class.
    auto class_const_inst = context_.insts().Get(
        context_.constant_values().GetInstId(class_const_id));
    if (class_const_inst.Is<SemIR::ClassType>()) {
      return {.const_id = class_const_id};
    } else {
      auto generic_class_type = context_.types().GetAs<SemIR::GenericClassType>(
          class_const_inst.type_id());
      auto instance_id = GetLocalGenericInstance(inst.instance_id, args);
      return ResolveAs<SemIR::ClassType>(
          {.type_id = SemIR::TypeId::TypeType,
           .class_id = generic_class_type.class_id,
           .instance_id = instance_id});
    }
  }

  auto TryResolveTypedInst(SemIR::ConstType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto inner_const_id = GetLocalConstantId(inst.inner_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto inner_type_id = context_.GetTypeIdForTypeConstant(inner_const_id);
    return ResolveAs<SemIR::ConstType>(
        {.type_id = SemIR::TypeId::TypeType, .inner_id = inner_type_id});
  }

  auto TryResolveTypedInst(SemIR::ExportDecl inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto value_id = GetLocalConstantId(inst.value_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    return {.const_id = value_id};
  }

  auto TryResolveTypedInst(SemIR::FieldDecl inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto inst_id = context_.AddInstInNoBlock<SemIR::FieldDecl>(
        AddImportIRInst(import_inst_id),
        {.type_id = context_.GetTypeIdForTypeConstant(const_id),
         .name_id = GetLocalNameId(inst.name_id),
         .index = inst.index});
    return {.const_id = context_.constant_values().Get(inst_id)};
  }

  auto TryResolveTypedInst(SemIR::FunctionDecl inst) -> ResolveResult {
    auto initial_work = work_stack_.size();

    const auto& function = import_ir_.functions().Get(inst.function_id);
    auto return_type_const_id = SemIR::ConstantId::Invalid;
    if (function.return_storage_id.is_valid()) {
      return_type_const_id =
          GetLocalConstantId(function.GetDeclaredReturnType(import_ir_));
    }
    auto parent_scope_id = GetLocalNameScopeId(function.parent_scope_id);
    llvm::SmallVector<SemIR::ConstantId> implicit_param_const_ids =
        GetLocalParamConstantIds(function.implicit_param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> param_const_ids =
        GetLocalParamConstantIds(function.param_refs_id);

    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Add the function declaration.
    SemIR::FunctionDecl function_decl = {
        .type_id = SemIR::TypeId::Invalid,
        .function_id = SemIR::FunctionId::Invalid,
        .decl_block_id = SemIR::InstBlockId::Empty};
    // Prefer pointing diagnostics towards a definition.
    auto import_ir_inst_id = AddImportIRInst(function.definition_id.is_valid()
                                                 ? function.definition_id
                                                 : function.decl_id);
    auto function_decl_id = context_.AddPlaceholderInstInNoBlock(
        SemIR::LocIdAndInst(import_ir_inst_id, function_decl));
    // TODO: Implement import for generics.
    auto generic_id = GetLocalGeneric(function.generic_id);

    auto new_return_storage = SemIR::InstId::Invalid;
    if (function.return_storage_id.is_valid()) {
      // Recreate the return slot from scratch.
      // TODO: Once we import function definitions, we'll need to make sure we
      // use the same return storage variable in the declaration and definition.
      new_return_storage = context_.AddInstInNoBlock<SemIR::VarStorage>(
          AddImportIRInst(function.return_storage_id),
          {.type_id = context_.GetTypeIdForTypeConstant(return_type_const_id),
           .name_id = SemIR::NameId::ReturnSlot});
    }
    function_decl.function_id = context_.functions().Add(
        {.name_id = GetLocalNameId(function.name_id),
         .parent_scope_id = parent_scope_id,
         .decl_id = function_decl_id,
         .generic_id = generic_id,
         .first_param_node_id = Parse::NodeId::Invalid,
         .last_param_node_id = Parse::NodeId::Invalid,
         .implicit_param_refs_id = GetLocalParamRefsId(
             function.implicit_param_refs_id, implicit_param_const_ids),
         .param_refs_id =
             GetLocalParamRefsId(function.param_refs_id, param_const_ids),
         .return_storage_id = new_return_storage,
         .is_extern = function.is_extern,
         .builtin_function_kind = function.builtin_function_kind,
         .definition_id = function.definition_id.is_valid()
                              ? function_decl_id
                              : SemIR::InstId::Invalid});
    // TODO: Import this or recompute it.
    auto specific_id = SemIR::GenericInstanceId::Invalid;
    function_decl.type_id =
        context_.GetFunctionType(function_decl.function_id, specific_id);
    // Write the function ID into the FunctionDecl.
    context_.ReplaceInstBeforeConstantUse(function_decl_id, function_decl);
    return {.const_id = context_.constant_values().Get(function_decl_id)};
  }

  auto TryResolveTypedInst(SemIR::FunctionType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto fn_val_id = GetLocalConstantInstId(
        import_ir_.functions().Get(inst.function_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto fn_val = context_.insts().Get(fn_val_id);
    CARBON_CHECK(context_.types().Is<SemIR::FunctionType>(fn_val.type_id()));
    // TODO: Import the correct generic instance and build a function type
    // constant using it.
    return {.const_id = context_.types().GetConstantId(fn_val.type_id())};
  }

  auto TryResolveTypedInst(SemIR::GenericClassType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_val_id =
        GetLocalConstantInstId(import_ir_.classes().Get(inst.class_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto class_val = context_.insts().Get(class_val_id);
    CARBON_CHECK(
        context_.types().Is<SemIR::GenericClassType>(class_val.type_id()));
    return {.const_id = context_.types().GetConstantId(class_val.type_id())};
  }

  auto TryResolveTypedInst(SemIR::GenericInterfaceType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto interface_val_id = GetLocalConstantInstId(
        import_ir_.interfaces().Get(inst.interface_id).decl_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }
    auto interface_val = context_.insts().Get(interface_val_id);
    CARBON_CHECK(context_.types().Is<SemIR::GenericInterfaceType>(
        interface_val.type_id()));
    return {.const_id =
                context_.types().GetConstantId(interface_val.type_id())};
  }

  auto TryResolveTypedInst(SemIR::ImportRefLoaded /*inst*/,
                           SemIR::InstId inst_id) -> ResolveResult {
    auto initial_work = work_stack_.size();
    // Return the constant for the instruction of the imported constant.
    auto constant_id = import_ir_.constant_values().Get(inst_id);
    if (!constant_id.is_valid()) {
      return {.const_id = SemIR::ConstantId::Error};
    }
    if (!constant_id.is_constant()) {
      context_.TODO(inst_id,
                    "Non-constant ImportRefLoaded (comes up with var)");
      return {.const_id = SemIR::ConstantId::Error};
    }

    auto new_constant_id =
        GetLocalConstantId(import_ir_.constant_values().GetInstId(constant_id));
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return {.const_id = new_constant_id};
  }

  // Make a declaration of an interface. This is done as a separate step from
  // importing the interface definition in order to resolve cycles.
  auto MakeInterfaceDecl(const SemIR::Interface& import_interface)
      -> std::pair<SemIR::InterfaceId, SemIR::ConstantId> {
    SemIR::InterfaceDecl interface_decl = {
        .type_id = SemIR::TypeId::TypeType,
        .interface_id = SemIR::InterfaceId::Invalid,
        .decl_block_id = SemIR::InstBlockId::Empty};
    auto interface_decl_id =
        context_.AddPlaceholderInstInNoBlock(SemIR::LocIdAndInst(
            AddImportIRInst(import_interface.decl_id), interface_decl));

    // TODO: Support for importing generics.
    auto generic_id = GetLocalGeneric(import_interface.generic_id);
    // Start with an incomplete interface.
    interface_decl.interface_id = context_.interfaces().Add({
        .name_id = GetLocalNameId(import_interface.name_id),
        // These are set in the second pass once we've imported them. Import
        // enough of the parameter lists that we know whether this interface is
        // a generic interface and can build the right constant value for it.
        // TODO: Add a better way to represent a generic `Interface` prior to
        // importing the parameters.
        .parent_scope_id = SemIR::NameScopeId::Invalid,
        .generic_id = generic_id,
        .first_param_node_id = Parse::NodeId::Invalid,
        .last_param_node_id = Parse::NodeId::Invalid,
        .implicit_param_refs_id =
            import_interface.implicit_param_refs_id.is_valid()
                ? SemIR::InstBlockId::Empty
                : SemIR::InstBlockId::Invalid,
        .param_refs_id = import_interface.param_refs_id.is_valid()
                             ? SemIR::InstBlockId::Empty
                             : SemIR::InstBlockId::Invalid,
        .decl_id = interface_decl_id,
    });

    if (import_interface.is_generic()) {
      interface_decl.type_id =
          context_.GetGenericInterfaceType(interface_decl.interface_id);
    }

    // Write the interface ID into the InterfaceDecl.
    context_.ReplaceInstBeforeConstantUse(interface_decl_id, interface_decl);
    return {interface_decl.interface_id,
            context_.constant_values().Get(interface_decl_id)};
  }

  // Imports the definition for an interface that has been imported as a forward
  // declaration.
  auto AddInterfaceDefinition(const SemIR::Interface& import_interface,
                              SemIR::Interface& new_interface,
                              SemIR::InstId self_param_id) -> void {
    new_interface.scope_id = context_.name_scopes().Add(
        new_interface.decl_id, SemIR::NameId::Invalid,
        new_interface.parent_scope_id);
    auto& new_scope = context_.name_scopes().Get(new_interface.scope_id);
    const auto& import_scope =
        import_ir_.name_scopes().Get(import_interface.scope_id);

    // Push a block so that we can add scoped instructions to it.
    context_.inst_block_stack().Push();
    AddNameScopeImportRefs(import_scope, new_scope);
    new_interface.associated_entities_id =
        AddAssociatedEntities(import_interface.associated_entities_id);
    new_interface.body_block_id = context_.inst_block_stack().Pop();
    new_interface.self_param_id = self_param_id;

    CARBON_CHECK(import_scope.extended_scopes.empty())
        << "Interfaces don't currently have extended scopes to support.";
  }

  auto TryResolveTypedInst(SemIR::InterfaceDecl inst,
                           SemIR::ConstantId interface_const_id)
      -> ResolveResult {
    const auto& import_interface =
        import_ir_.interfaces().Get(inst.interface_id);

    SemIR::InterfaceId interface_id = SemIR::InterfaceId::Invalid;
    if (!interface_const_id.is_valid()) {
      // On the first pass, create a forward declaration of the interface.
      std::tie(interface_id, interface_const_id) =
          MakeInterfaceDecl(import_interface);
    } else {
      // On the second pass, compute the interface ID from the constant value of
      // the declaration.
      auto interface_const_inst = context_.insts().Get(
          context_.constant_values().GetInstId(interface_const_id));
      if (auto interface_type =
              interface_const_inst.TryAs<SemIR::InterfaceType>()) {
        interface_id = interface_type->interface_id;
      } else {
        auto generic_interface_type =
            context_.types().GetAs<SemIR::GenericInterfaceType>(
                interface_const_inst.type_id());
        interface_id = generic_interface_type.interface_id;
      }
    }

    auto initial_work = work_stack_.size();

    auto parent_scope_id =
        GetLocalNameScopeId(import_interface.parent_scope_id);
    llvm::SmallVector<SemIR::ConstantId> implicit_param_const_ids =
        GetLocalParamConstantIds(import_interface.implicit_param_refs_id);
    llvm::SmallVector<SemIR::ConstantId> param_const_ids =
        GetLocalParamConstantIds(import_interface.param_refs_id);

    std::optional<SemIR::InstId> self_param_id;
    if (import_interface.is_defined()) {
      self_param_id = GetLocalConstantInstId(import_interface.self_param_id);
    }

    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry(interface_const_id);
    }

    auto& new_interface = context_.interfaces().Get(interface_id);
    new_interface.parent_scope_id = parent_scope_id;
    new_interface.implicit_param_refs_id = GetLocalParamRefsId(
        import_interface.implicit_param_refs_id, implicit_param_const_ids);
    new_interface.param_refs_id =
        GetLocalParamRefsId(import_interface.param_refs_id, param_const_ids);

    if (import_interface.is_defined()) {
      CARBON_CHECK(self_param_id);
      AddInterfaceDefinition(import_interface, new_interface, *self_param_id);
    }
    return {.const_id = interface_const_id};
  }

  auto TryResolveTypedInst(SemIR::InterfaceType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto interface_const_id = GetLocalConstantId(
        import_ir_.interfaces().Get(inst.interface_id).decl_id);
    auto args = GetLocalGenericInstanceArgs(inst.instance_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    // Find the corresponding interface type. For a non-generic interface, this
    // is the type of the interface declaration. For a generic interface, build
    // a interface type referencing this specialization of the generic
    // interface.
    auto interface_const_inst = context_.insts().Get(
        context_.constant_values().GetInstId(interface_const_id));
    if (interface_const_inst.Is<SemIR::InterfaceType>()) {
      return {.const_id = interface_const_id};
    } else {
      auto generic_interface_type =
          context_.types().GetAs<SemIR::GenericInterfaceType>(
              interface_const_inst.type_id());
      auto instance_id = GetLocalGenericInstance(inst.instance_id, args);
      return ResolveAs<SemIR::InterfaceType>(
          {.type_id = SemIR::TypeId::TypeType,
           .interface_id = generic_interface_type.interface_id,
           .instance_id = instance_id});
    }
  }

  auto TryResolveTypedInst(SemIR::InterfaceWitness inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto elements = GetLocalInstBlockContents(inst.elements_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    auto elements_id = GetLocalCanonicalInstBlockId(inst.elements_id, elements);
    return ResolveAs<SemIR::InterfaceWitness>(
        {.type_id =
             context_.GetBuiltinType(SemIR::BuiltinInstKind::WitnessType),
         .elements_id = elements_id});
  }

  auto TryResolveTypedInst(SemIR::IntLiteral inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return ResolveAs<SemIR::IntLiteral>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .int_id = context_.ints().Add(import_ir_.ints().Get(inst.int_id))});
  }

  auto TryResolveTypedInst(SemIR::PointerType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto pointee_const_id = GetLocalConstantId(inst.pointee_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    auto pointee_type_id = context_.GetTypeIdForTypeConstant(pointee_const_id);
    return ResolveAs<SemIR::PointerType>(
        {.type_id = SemIR::TypeId::TypeType, .pointee_id = pointee_type_id});
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
      fields.push_back(context_.AddInstInNoBlock<SemIR::StructTypeField>(
          AddImportIRInst(import_inst_id),
          {.name_id = name_id, .field_type_id = field_type_id}));
    }

    return ResolveAs<SemIR::StructType>(
        {.type_id = SemIR::TypeId::TypeType,
         .fields_id = context_.inst_blocks().AddCanonical(fields)});
  }

  auto TryResolveTypedInst(SemIR::StructValue inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_id = GetLocalConstantId(inst.type_id);
    auto elems = GetLocalInstBlockContents(inst.elements_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return ResolveAs<SemIR::StructValue>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .elements_id = GetLocalCanonicalInstBlockId(inst.elements_id, elems)});
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

    return {.const_id = context_.types().GetConstantId(
                context_.GetTupleType(elem_type_ids))};
  }

  auto TryResolveTypedInst(SemIR::TupleValue inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    auto type_id = GetLocalConstantId(inst.type_id);
    auto elems = GetLocalInstBlockContents(inst.elements_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return ResolveAs<SemIR::TupleValue>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .elements_id = GetLocalCanonicalInstBlockId(inst.elements_id, elems)});
  }

  auto TryResolveTypedInst(SemIR::UnboundElementType inst) -> ResolveResult {
    auto initial_work = work_stack_.size();
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id = GetLocalConstantId(inst.class_type_id);
    auto elem_const_id = GetLocalConstantId(inst.element_type_id);
    if (HasNewWork(initial_work)) {
      return ResolveResult::Retry();
    }

    return ResolveAs<SemIR::UnboundElementType>(
        {.type_id = SemIR::TypeId::TypeType,
         .class_type_id = context_.GetTypeIdForTypeConstant(class_const_id),
         .element_type_id = context_.GetTypeIdForTypeConstant(elem_const_id)});
  }

  auto import_ir_constant_values() -> SemIR::ConstantValueStore& {
    return context_.import_ir_constant_values()[import_ir_id_.index];
  }

  Context& context_;
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File& import_ir_;
  llvm::SmallVector<Work> work_stack_;
};

// Returns a list of ImportIRInsts equivalent to the ImportRef currently being
// loaded (including the one pointed at directly by the ImportRef), and the
// final instruction's type ID.
//
// This addresses cases where an ImportRefUnloaded may point at another
// ImportRefUnloaded. The ImportRefResolver requires a SemIR with a
// constant-evaluated version of the instruction to work with.
static auto GetInstForLoad(Context& context,
                           SemIR::ImportIRInstId import_ir_inst_id)
    -> std::pair<llvm::SmallVector<SemIR::ImportIRInst>, SemIR::TypeId> {
  std::pair<llvm::SmallVector<SemIR::ImportIRInst>, SemIR::TypeId> result = {
      {}, SemIR::TypeId::Invalid};
  auto& [import_ir_insts, type_id] = result;

  auto import_ir_inst = context.import_ir_insts().Get(import_ir_inst_id);
  // The first ImportIRInst is added directly because the IR doesn't need to be
  // localized.
  import_ir_insts.push_back(import_ir_inst);
  const auto* cursor_ir = context.import_irs().Get(import_ir_inst.ir_id).sem_ir;

  while (true) {
    auto cursor_inst = cursor_ir->insts().Get(import_ir_inst.inst_id);

    auto import_ref = cursor_inst.TryAs<SemIR::ImportRefUnloaded>();
    if (!import_ref) {
      type_id = cursor_inst.type_id();
      return result;
    }

    import_ir_inst =
        cursor_ir->import_ir_insts().Get(import_ref->import_ir_inst_id);
    cursor_ir = cursor_ir->import_irs().Get(import_ir_inst.ir_id).sem_ir;
    import_ir_insts.push_back(
        {.ir_id = AddImportIR(context, {.decl_id = SemIR::InstId::Invalid,
                                        .is_export = false,
                                        .sem_ir = cursor_ir}),
         .inst_id = import_ir_inst.inst_id});
  }
}

auto LoadImportRef(Context& context, SemIR::InstId inst_id) -> void {
  auto inst = context.insts().TryGetAs<SemIR::ImportRefUnloaded>(inst_id);
  if (!inst) {
    return;
  }

  auto [indirect_insts, load_type_id] =
      GetInstForLoad(context, inst->import_ir_inst_id);

  // The last indirect instruction is the one to resolve. Pop it here because
  // Resolve will assign the constant.
  auto load_ir_inst = indirect_insts.pop_back_val();
  ImportRefResolver resolver(context, load_ir_inst.ir_id);
  // The resolver calls into Context to create instructions. Don't register
  // those instructions as part of the enclosing generic scope if they're
  // dependent on a generic parameter.
  context.generic_region_stack().Push();
  auto type_id = resolver.ResolveType(load_type_id);
  auto constant_id = resolver.Resolve(load_ir_inst.inst_id);
  context.generic_region_stack().Pop();

  // Replace the ImportRefUnloaded instruction with ImportRefLoaded. This
  // doesn't use ReplaceInstBeforeConstantUse because it would trigger
  // TryEvalInst, which we want to avoid with ImportRefs.
  context.sem_ir().insts().Set(
      inst_id,
      SemIR::ImportRefLoaded{.type_id = type_id,
                             .import_ir_inst_id = inst->import_ir_inst_id,
                             .entity_name_id = inst->entity_name_id});

  // Store the constant for both the ImportRefLoaded and indirect instructions.
  context.constant_values().Set(inst_id, constant_id);
  for (const auto& import_ir_inst : indirect_insts) {
    context.import_ir_constant_values()[import_ir_inst.ir_id.index].Set(
        import_ir_inst.inst_id, constant_id);
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
    impl.witness_id = AddImportRef(
        context, {.ir_id = import_ir_id, .inst_id = import_impl.witness_id},
        SemIR::EntityNameId::Invalid);
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
