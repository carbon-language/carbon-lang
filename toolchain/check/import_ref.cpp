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
#include "toolchain/sem_ir/type_info.h"
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
  CARBON_CHECK(ir_id == SemIR::ImportIRId::ApiForImpl,
               "ApiForImpl must be the first IR");
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
                  SemIR::EntityNameId entity_name_id =
                      SemIR::EntityNameId::Invalid) -> SemIR::InstId {
  auto import_ir_inst_id = context.import_ir_insts().Add(import_ir_inst);
  SemIR::ImportRefUnloaded inst = {.import_ir_inst_id = import_ir_inst_id,
                                   .entity_name_id = entity_name_id};
  auto import_ref_id = context.AddPlaceholderInstInNoBlock(
      context.MakeImportedLocAndInst(import_ir_inst_id, inst));

  // ImportRefs have a dedicated block because this may be called during
  // processing where the instruction shouldn't be inserted in the current inst
  // block.
  context.import_ref_ids().push_back(import_ref_id);
  return import_ref_id;
}

// Adds an import_ref instruction for an instruction that we have already loaded
// from an imported IR, with a known constant value. This is useful when the
// instruction has a symbolic constant value, in order to produce an instruction
// that hold that symbolic constant.
static auto AddLoadedImportRef(Context& context,
                               SemIR::ImportIRInst import_ir_inst,
                               SemIR::TypeId type_id,
                               SemIR::ConstantId const_id) -> SemIR::InstId {
  auto import_ir_inst_id = context.import_ir_insts().Add(import_ir_inst);
  SemIR::ImportRefLoaded inst = {
      .type_id = type_id,
      .import_ir_inst_id = import_ir_inst_id,
      .entity_name_id = SemIR::EntityNameId::Invalid};
  auto inst_id = context.AddPlaceholderInstInNoBlock(
      context.MakeImportedLocAndInst(import_ir_inst_id, inst));
  context.import_ref_ids().push_back(inst_id);

  context.constant_values().Set(inst_id, const_id);
  context.import_ir_constant_values()[import_ir_inst.ir_id.index].Set(
      import_ir_inst.inst_id, const_id);
  return inst_id;
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
      AddImportRef(context, {.ir_id = new_ir_id, .inst_id = new_inst_id});
  context.DiagnoseDuplicateName(conflict_id, prev_id);
}

// Returns an instruction that has the specified constant value.
static auto GetInstWithConstantValue(const SemIR::File& file,
                                     SemIR::ConstantId const_id)
    -> SemIR::InstId {
  if (!const_id.is_valid()) {
    return SemIR::InstId::Invalid;
  }

  // For template constants, the corresponding instruction has the desired
  // constant value.
  if (!const_id.is_symbolic()) {
    return file.constant_values().GetInstId(const_id);
  }

  // For abstract symbolic constants, the corresponding instruction has the
  // desired constant value.
  const auto& symbolic_const =
      file.constant_values().GetSymbolicConstant(const_id);
  if (!symbolic_const.generic_id.is_valid()) {
    return file.constant_values().GetInstId(const_id);
  }

  // For a symbolic constant in a generic, pick the corresponding instruction
  // out of the eval block for the generic.
  const auto& generic = file.generics().Get(symbolic_const.generic_id);
  auto block = generic.GetEvalBlock(symbolic_const.index.region());
  return file.inst_blocks().Get(block)[symbolic_const.index.index()];
}

// Resolves an instruction from an imported IR into a constant referring to the
// current IR.
//
// Calling Resolve on an instruction operates in an iterative manner, tracking
// Work items on work_stack_. At a high level, the loop is:
//
// 1. If a constant value is already known for the work item and was not set by
//    this work item, it's considered resolved.
//    - The constant check avoids performance costs of deduplication on add.
//    - If we've processed this work item before, then we now process it again.
//      It didn't complete last time, even though we have a constant value
//      already.
//
// 2. Resolve the instruction (TryResolveInst/TryResolveTypedInst). This is done
//    in three phases. The first and second phases can add work to the worklist
//    and end in a retry, in which case those phases will be rerun once the
//    added work is done. The rerun cannot also end in a retry, so this results
//    in at most three calls, but in practice one or two calls is almost always
//    sufficient. Due to the chance of a second or third call to TryResolveInst,
//    it's important to only perform expensive work once, even when the same
//    phase is rerun.
//
//    - First phase:
//      - Gather all input constants necessary to form the constant value of the
//        instruction. Gathering constants directly adds unresolved values to
//        work_stack_.
//      - If HasNewWork() reports that any work was added, then return Retry():
//        this instruction needs another call to complete. Gather the
//        now-resolved constants and continue to the next step once the retry
//        happens.
//
//    - Second phase:
//      - Build the constant value of the instruction.
//      - Gather all input constants necessary to finish importing the
//        instruction. This is only necessary for instructions like classes that
//        can be forward-declared. For these instructions, we first import the
//        constant value and then later import the rest of the declaration in
//        order to break cycles.
//      - If HasNewWork() reports that any work was added, then return
//        Retry(constant_value): this instruction needs another call to
//        complete.  Gather the now-resolved constants and continue to the next
//        step once the retry happens.
//
//    - Third phase:
//      - After the second phase, the constant value for the instruction is
//        already set, and will be passed back into TryResolve*Inst on retry. It
//        should not be created again.
//      - Fill in any remaining information to complete the import of the
//        instruction. For example, when importing a class declaration, build
//        the class scope and information about the definition.
//      - Return ResolveAs/ResolveAsConstant to finish the resolution process.
//        This will cause the Resolve loop to set a constant value if we didn't
//        retry at the end of the second phase.
//
// 3. If resolve didn't return Retry(), pop the work. Otherwise, it needs to
//    remain, and may no longer be at the top of the stack; update the state of
//    the work item to track what work still needs to be done.
//
// The same instruction can be enqueued for resolution multiple times. However,
// we will only reach the second phase once: once a constant value is set, only
// the resolution step that set it will retry.
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
  auto ResolveOneInst(SemIR::InstId inst_id) -> SemIR::ConstantId {
    work_stack_.push_back({.inst_id = inst_id});
    while (!work_stack_.empty()) {
      auto work = work_stack_.back();
      CARBON_CHECK(work.inst_id.is_valid());

      // Step 1: check for a constant value.
      auto existing = FindResolvedConstId(work.inst_id);
      if (existing.const_id.is_valid() && !work.retry_with_constant_value) {
        work_stack_.pop_back();
        continue;
      }

      // Step 2: resolve the instruction.
      initial_work_ = work_stack_.size();
      auto [new_const_id, retry] =
          TryResolveInst(work.inst_id, existing.const_id);

      CARBON_CHECK(
          !existing.const_id.is_valid() || existing.const_id == new_const_id,
          "Constant value changed in third phase.");
      if (!existing.const_id.is_valid()) {
        SetResolvedConstId(work.inst_id, existing.indirect_insts, new_const_id);
      }

      // Step 3: pop or retry.
      if (retry) {
        work_stack_[initial_work_ - 1].retry_with_constant_value =
            new_const_id.is_valid();
      } else {
        work_stack_.pop_back();
      }
    }
    auto constant_id = import_ir_constant_values().Get(inst_id);
    CARBON_CHECK(constant_id.is_valid());
    return constant_id;
  }

  // Performs resolution for one instruction and then performs all work we
  // deferred.
  auto Resolve(SemIR::InstId inst_id) -> SemIR::ConstantId {
    auto const_id = ResolveOneInst(inst_id);
    PerformPendingWork();
    return const_id;
  }

  // Wraps constant evaluation with logic to handle constants.
  auto ResolveConstant(SemIR::ConstantId import_const_id) -> SemIR::ConstantId {
    return Resolve(GetInstWithConstantValue(import_ir_, import_const_id));
  }

  // Wraps constant evaluation with logic to handle types.
  auto ResolveType(SemIR::TypeId import_type_id) -> SemIR::TypeId {
    if (!import_type_id.is_valid()) {
      return import_type_id;
    }

    auto import_type_const_id =
        import_ir_.types().GetConstantId(import_type_id);
    CARBON_CHECK(import_type_const_id.is_valid());

    if (auto import_type_inst_id =
            import_ir_.constant_values().GetInstId(import_type_const_id);
        import_type_inst_id.is_builtin()) {
      // Builtins don't require constant resolution; we can use them directly.
      return context_.GetBuiltinType(import_type_inst_id.builtin_inst_kind());
    } else {
      return context_.GetTypeIdForTypeConstant(
          ResolveConstant(import_type_id.AsConstantId()));
    }
  }

 private:
  // The result of attempting to resolve an imported instruction to a constant.
  struct ResolveResult {
    // The new constant value, if known.
    SemIR::ConstantId const_id;
    // Whether resolution has been attempted once and needs to be retried.
    bool retry = false;
  };

  // A step in work_stack_.
  struct Work {
    // The instruction to work on.
    SemIR::InstId inst_id;
    // Whether this work item set the constant value for the instruction and
    // requested a retry.
    bool retry_with_constant_value = false;
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

  // Local information associated with an imported generic.
  struct GenericData {
    llvm::SmallVector<SemIR::InstId> bindings;
  };

  // Local information associated with an imported specific.
  struct SpecificData {
    SemIR::ConstantId generic_const_id;
    llvm::SmallVector<SemIR::InstId> args;
  };

  // A generic that we have partially imported.
  struct PendingGeneric {
    SemIR::GenericId import_id;
    SemIR::GenericId local_id;
  };

  // A specific that we have partially imported.
  struct PendingSpecific {
    SemIR::SpecificId import_id;
    SemIR::SpecificId local_id;
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

      CARBON_CHECK(cursor_ir != prev_ir || cursor_inst_id != prev_inst_id,
                   "{0}", cursor_ir->insts().Get(cursor_inst_id));

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

  // Returns true if new unresolved constants were found as part of this
  // `Resolve` step.
  auto HasNewWork() -> bool {
    CARBON_CHECK(initial_work_ <= work_stack_.size(),
                 "Work shouldn't decrease");
    return initial_work_ < work_stack_.size();
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

  // Returns the ConstantId for an imported ConstantId. Adds unresolved
  // constants to work_stack_.
  auto GetLocalConstantId(SemIR::ConstantId const_id) -> SemIR::ConstantId {
    return GetLocalConstantId(GetInstWithConstantValue(import_ir_, const_id));
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
    return GetLocalConstantId(import_ir_.types().GetConstantId(type_id));
  }

  template <typename Id>
  auto GetLocalConstantIdChecked(Id id) {
    auto result = GetLocalConstantId(id);
    CARBON_CHECK(result.is_valid());
    return result;
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

  // Gets a local instruction block ID corresponding to an imported inst block
  // whose contents were already imported, for example by
  // GetLocalInstBlockContents.
  auto GetLocalInstBlockId(SemIR::InstBlockId import_block_id,
                           llvm::ArrayRef<SemIR::InstId> contents)
      -> SemIR::InstBlockId {
    if (!import_block_id.is_valid()) {
      return SemIR::InstBlockId::Invalid;
    }
    return context_.inst_blocks().Add(contents);
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

  // Gets an incomplete local version of an imported generic. Most fields are
  // set in the third phase.
  auto MakeIncompleteGeneric(SemIR::InstId decl_id, SemIR::GenericId generic_id)
      -> SemIR::GenericId {
    if (!generic_id.is_valid()) {
      return SemIR::GenericId::Invalid;
    }

    return context_.generics().Add(
        {.decl_id = decl_id,
         .bindings_id = SemIR::InstBlockId::Invalid,
         .self_specific_id = SemIR::SpecificId::Invalid});
  }

  // Gets a local version of the data associated with a generic.
  auto GetLocalGenericData(SemIR::GenericId generic_id) -> GenericData {
    if (!generic_id.is_valid()) {
      return GenericData();
    }

    const auto& generic = import_ir_.generics().Get(generic_id);
    return {.bindings = GetLocalInstBlockContents(generic.bindings_id)};
  }

  // Adds the given local generic data to the given generic.
  auto SetGenericData(SemIR::GenericId import_generic_id,
                      SemIR::GenericId new_generic_id,
                      const GenericData& generic_data) -> void {
    if (!import_generic_id.is_valid()) {
      return;
    }

    const auto& import_generic = import_ir_.generics().Get(import_generic_id);
    auto& new_generic = context_.generics().Get(new_generic_id);
    new_generic.bindings_id = GetLocalCanonicalInstBlockId(
        import_generic.bindings_id, generic_data.bindings);
    // Fill in the remaining information in FinishPendingGeneric.
    pending_generics_.push_back(
        {.import_id = import_generic_id, .local_id = new_generic_id});
  }

  // Gets a local constant value corresponding to an imported generic ID. May
  // add work to the work stack and return `Invalid`.
  auto GetLocalConstantId(SemIR::GenericId generic_id) -> SemIR::ConstantId {
    if (!generic_id.is_valid()) {
      return SemIR::ConstantId::Invalid;
    }
    auto import_decl_inst_id = import_ir_.generics().Get(generic_id).decl_id;
    auto import_decl_inst = import_ir_.insts().Get(import_decl_inst_id);
    if (import_decl_inst.Is<SemIR::ImplDecl>()) {
      // For an impl declaration, the imported entity can be found via the
      // declaration.
      return GetLocalConstantId(import_decl_inst_id);
    }
    // For all other kinds of declaration, the imported entity can be found via
    // the type of the declaration.
    return GetLocalConstantId(import_decl_inst.type_id());
  }

  // Gets a local generic ID given the corresponding local constant ID returned
  // by GetLocalConstantId for the imported generic. Does not add any new work.
  auto GetLocalGenericId(SemIR::ConstantId local_const_id) -> SemIR::GenericId {
    if (!local_const_id.is_valid()) {
      return SemIR::GenericId::Invalid;
    }
    auto inst = context_.insts().Get(
        context_.constant_values().GetInstId(local_const_id));
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(SemIR::FunctionType fn_type): {
        return context_.functions().Get(fn_type.function_id).generic_id;
      }
      case CARBON_KIND(SemIR::GenericClassType class_type): {
        return context_.classes().Get(class_type.class_id).generic_id;
      }
      case CARBON_KIND(SemIR::GenericInterfaceType interface_type): {
        return context_.interfaces()
            .Get(interface_type.interface_id)
            .generic_id;
      }
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        return context_.impls().Get(impl_decl.impl_id).generic_id;
      }
      default: {
        CARBON_FATAL("Unexpected inst for generic declaration: {0}", inst);
      }
    }
  }

  // Gets local information about an imported specific.
  auto GetLocalSpecificData(SemIR::SpecificId specific_id) -> SpecificData {
    if (!specific_id.is_valid()) {
      return {.generic_const_id = SemIR::ConstantId::Invalid, .args = {}};
    }

    const auto& specific = import_ir_.specifics().Get(specific_id);
    return {
        .generic_const_id = GetLocalConstantId(specific.generic_id),
        .args = GetLocalInstBlockContents(specific.args_id),
    };
  }

  // Gets a local specific whose data was already imported by
  // GetLocalSpecificData. Does not add any new work.
  auto GetOrAddLocalSpecific(SemIR::SpecificId import_specific_id,
                             const SpecificData& data) -> SemIR::SpecificId {
    if (!import_specific_id.is_valid()) {
      return SemIR::SpecificId::Invalid;
    }

    // Form a corresponding local specific ID.
    const auto& import_specific =
        import_ir_.specifics().Get(import_specific_id);
    auto generic_id = GetLocalGenericId(data.generic_const_id);
    auto args_id =
        GetLocalCanonicalInstBlockId(import_specific.args_id, data.args);

    // Get the specific.
    auto specific_id = context_.specifics().GetOrAdd(generic_id, args_id);

    // Fill in the remaining information in FinishPendingSpecific, if necessary.
    auto& specific = context_.specifics().Get(specific_id);
    if (!specific.decl_block_id.is_valid() ||
        (import_specific.definition_block_id.is_valid() &&
         !specific.definition_block_id.is_valid())) {
      pending_specifics_.push_back(
          {.import_id = import_specific_id, .local_id = specific_id});
    }
    return specific_id;
  }

  // Adds unresolved constants for each parameter's type to work_stack_.
  auto LoadLocalParamConstantIds(SemIR::InstBlockId param_refs_id) -> void {
    if (!param_refs_id.is_valid() ||
        param_refs_id == SemIR::InstBlockId::Empty) {
      return;
    }

    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
    for (auto inst_id : param_refs) {
      GetLocalConstantId(import_ir_.insts().Get(inst_id).type_id());

      // If the parameter is a symbolic binding, build the BindSymbolicName
      // constant.
      auto bind_id = inst_id;
      auto bind_inst = import_ir_.insts().Get(bind_id);
      if (auto addr = bind_inst.TryAs<SemIR::AddrPattern>()) {
        bind_id = addr->inner_id;
        bind_inst = import_ir_.insts().Get(bind_id);
      }
      if (bind_inst.Is<SemIR::BindSymbolicName>()) {
        GetLocalConstantId(bind_id);
      }
    }
  }

  // Returns a version of param_refs_id localized to the current IR.
  //
  // Must only be called after a call to GetLocalParamConstantIds(param_refs_id)
  // has completed without adding any new work to work_stack_.
  //
  // TODO: This is inconsistent with the rest of this class, which expects
  // the relevant constants to be explicitly passed in. That makes it
  // easier to statically detect when an input isn't loaded, but makes it
  // harder to support importing more complex inst structures. We should
  // take a holistic look at how to balance those concerns. For example,
  // could the same function be used to load the constants and use them, with
  // a parameter to select between the two?
  auto GetLocalParamRefsId(SemIR::InstBlockId param_refs_id)
      -> SemIR::InstBlockId {
    if (!param_refs_id.is_valid() ||
        param_refs_id == SemIR::InstBlockId::Empty) {
      return param_refs_id;
    }
    const auto& param_refs = import_ir_.inst_blocks().Get(param_refs_id);
    llvm::SmallVector<SemIR::InstId> new_param_refs;
    for (auto ref_id : param_refs) {
      // Figure out the param structure. This echoes
      // Function::GetParamFromParamRefId, and could use that function if we
      // added `bool addr` and `InstId bind_inst_id` to its return `ParamInfo`.
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
      auto type_id = context_.GetTypeIdForTypeConstant(
          GetLocalConstantIdChecked(param_inst.type_id));

      auto new_param_id = context_.AddInstInNoBlock<SemIR::Param>(
          AddImportIRInst(param_id),
          {.type_id = type_id,
           .name_id = name_id,
           .runtime_index = param_inst.runtime_index});
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
            // We already imported a constant value for this symbolic binding.
            // We can reuse most of it, but update the value to point to our
            // specific parameter, and preserve the constant value.
            auto new_bind_inst =
                context_.insts().GetAs<SemIR::BindSymbolicName>(
                    context_.constant_values().GetInstId(
                        GetLocalConstantIdChecked(bind_id)));
            new_bind_inst.value_id = new_param_id;
            new_param_id = context_.AddInstInNoBlock(AddImportIRInst(bind_id),
                                                     new_bind_inst);
            context_.constant_values().Set(new_param_id,
                                           GetLocalConstantIdChecked(bind_id));
            break;
          }
          default: {
            CARBON_FATAL("Unexpected kind: {0}", bind_inst->kind);
          }
        }
      }
      if (addr_inst) {
        new_param_id = context_.AddInstInNoBlock(
            context_.MakeImportedLocAndInst<SemIR::AddrPattern>(
                AddImportIRInst(ref_id),
                {.type_id = type_id, .inner_id = new_param_id}));
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
      case CARBON_KIND(SemIR::ImplDecl inst): {
        return context_.impls().Get(inst.impl_id).scope_id;
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
    CARBON_FATAL("Unexpected instruction kind for name scope: {0}",
                 name_scope_inst);
  }

  // Given an imported entity base, returns an incomplete, local version of it.
  //
  // Most fields are set in the third phase once they're imported. Import enough
  // of the parameter lists that we know whether this interface is a generic
  // interface and can build the right constant value for it.
  //
  // TODO: Support extern.
  // TODO: Add a better way to represent a generic prior to importing the
  // parameters.
  auto GetIncompleteLocalEntityBase(
      SemIR::InstId decl_id, const SemIR::EntityWithParamsBase& import_base)
      -> SemIR::EntityWithParamsBase {
    // Translate the extern_library_id if present.
    auto extern_library_id = SemIR::LibraryNameId::Invalid;
    if (import_base.extern_library_id.is_valid()) {
      if (import_base.extern_library_id.index >= 0) {
        auto val = import_ir_.string_literal_values().Get(
            import_base.extern_library_id.AsStringLiteralValueId());
        extern_library_id = SemIR::LibraryNameId::ForStringLiteralValueId(
            context_.string_literal_values().Add(val));
      } else {
        extern_library_id = import_base.extern_library_id;
      }
    }

    return {
        .name_id = GetLocalNameId(import_base.name_id),
        .parent_scope_id = SemIR::NameScopeId::Invalid,
        .generic_id = MakeIncompleteGeneric(decl_id, import_base.generic_id),
        .first_param_node_id = Parse::NodeId::Invalid,
        .last_param_node_id = Parse::NodeId::Invalid,
        .pattern_block_id = SemIR::InstBlockId::Invalid,
        .implicit_param_refs_id = import_base.implicit_param_refs_id.is_valid()
                                      ? SemIR::InstBlockId::Empty
                                      : SemIR::InstBlockId::Invalid,
        .param_refs_id = import_base.param_refs_id.is_valid()
                             ? SemIR::InstBlockId::Empty
                             : SemIR::InstBlockId::Invalid,
        .is_extern = import_base.is_extern,
        .extern_library_id = extern_library_id,
        .non_owning_decl_id = import_base.non_owning_decl_id.is_valid()
                                  ? decl_id
                                  : SemIR::InstId::Invalid,
        .first_owning_decl_id = import_base.first_owning_decl_id.is_valid()
                                    ? decl_id
                                    : SemIR::InstId::Invalid,
    };
  }

  // Adds ImportRefUnloaded entries for members of the imported scope, for name
  // lookup.
  auto AddNameScopeImportRefs(const SemIR::NameScope& import_scope,
                              SemIR::NameScope& new_scope) -> void {
    for (auto entry : import_scope.names) {
      auto ref_id = AddImportRef(
          context_, {.ir_id = import_ir_id_, .inst_id = entry.inst_id});
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
          AddImportRef(context_, {.ir_id = import_ir_id_, .inst_id = inst_id}));
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
    auto inst_const_id = import_ir_.constant_values().Get(inst_id);
    if (!inst_const_id.is_valid() || !inst_const_id.is_symbolic()) {
      return TryResolveInstCanonical(inst_id, const_id);
    }

    // Try to import the generic. This might add new work.
    const auto& symbolic_const =
        import_ir_.constant_values().GetSymbolicConstant(inst_const_id);
    auto generic_const_id = GetLocalConstantId(symbolic_const.generic_id);

    auto inner_const_id = SemIR::ConstantId::Invalid;
    if (const_id.is_valid()) {
      // For the third phase, extract the constant value that
      // TryResolveInstCanonical produced previously.
      inner_const_id = context_.constant_values().Get(
          context_.constant_values().GetSymbolicConstant(const_id).inst_id);
    }

    // Import the constant and rebuild the symbolic constant data.
    auto result = TryResolveInstCanonical(inst_id, inner_const_id);
    if (!result.const_id.is_valid()) {
      // First phase: TryResolveInstCanoncial needs a retry.
      return result;
    }

    if (!const_id.is_valid()) {
      // Second phase: we have created an abstract constant. Create a
      // corresponding generic constant.
      if (symbolic_const.generic_id.is_valid()) {
        result.const_id = context_.constant_values().AddSymbolicConstant(
            {.inst_id = context_.constant_values().GetInstId(result.const_id),
             .generic_id = GetLocalGenericId(generic_const_id),
             .index = symbolic_const.index});
      }
    } else {
      // Third phase: perform a consistency check and produce the constant we
      // created in the second phase.
      CARBON_CHECK(result.const_id == inner_const_id,
                   "Constant value changed in third phase.");
      result.const_id = const_id;
    }

    return result;
  }

  // Tries to resolve the InstId, returning a canonical constant when ready, or
  // Invalid if more has been added to the stack. This is the same as
  // TryResolveInst, except that it may resolve symbolic constants as canonical
  // constants instead of as constants associated with a particular generic.
  auto TryResolveInstCanonical(SemIR::InstId inst_id,
                               SemIR::ConstantId const_id) -> ResolveResult {
    if (inst_id.is_builtin()) {
      CARBON_CHECK(!const_id.is_valid());
      // Constants for builtins can be directly copied.
      return ResolveAsConstant(context_.constant_values().Get(inst_id));
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
      case SemIR::BindName::Kind: {
        // TODO: Should we be resolving BindNames at all?
        return ResolveAsConstant(SemIR::ConstantId::NotConstant);
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
      case CARBON_KIND(SemIR::CompleteTypeWitness inst): {
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
        return TryResolveTypedInst(inst, const_id);
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
      case CARBON_KIND(SemIR::ImplDecl inst): {
        return TryResolveTypedInst(inst, const_id);
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

  // Produces a resolve result that tries resolving this instruction again. If
  // `const_id` is specified, then this is the end of the second phase, and the
  // constant value will be passed to the next resolution attempt. Otherwise,
  // this is the end of the first phase.
  auto Retry(SemIR::ConstantId const_id = SemIR::ConstantId::Invalid)
      -> ResolveResult {
    CARBON_CHECK(HasNewWork());
    return {.const_id = const_id, .retry = true};
  }

  // Produces a resolve result that provides the given constant value. Requires
  // that there is no new work.
  auto ResolveAsConstant(SemIR::ConstantId const_id) -> ResolveResult {
    CARBON_CHECK(!HasNewWork());
    return {.const_id = const_id};
  }

  // Produces a resolve result that provides the given constant value. Retries
  // instead if work has been added.
  auto RetryOrResolveAsConstant(SemIR::ConstantId const_id) -> ResolveResult {
    if (HasNewWork()) {
      return Retry();
    }
    return ResolveAsConstant(const_id);
  }

  // Produces a resolve result for the given instruction that describes a
  // constant value. This should only be used for instructions that describe
  // constants, and not for instructions that represent declarations. For a
  // declaration, we need an associated location, so AddInstInNoBlock should be
  // used instead. Requires that there is no new work.
  auto ResolveAsUntyped(SemIR::Inst inst) -> ResolveResult {
    CARBON_CHECK(!HasNewWork());
    auto result = TryEvalInst(context_, SemIR::InstId::Invalid, inst);
    CARBON_CHECK(result.is_constant(), "{0} is not constant", inst);
    return {.const_id = result};
  }

  // Same as ResolveAsUntyped, but with an explicit type for convenience.
  template <typename InstT>
  auto ResolveAs(InstT inst) -> ResolveResult {
    return ResolveAsUntyped(inst);
  }

  auto TryResolveTypedInst(SemIR::AssociatedEntity inst) -> ResolveResult {
    auto type_const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork()) {
      return Retry();
    }

    // Add a lazy reference to the target declaration.
    auto decl_id = AddImportRef(
        context_, {.ir_id = import_ir_id_, .inst_id = inst.decl_id});

    return ResolveAs<SemIR::AssociatedEntity>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_const_id),
         .index = inst.index,
         .decl_id = decl_id});
  }

  auto TryResolveTypedInst(SemIR::AssociatedEntityType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    auto entity_type_const_id = GetLocalConstantId(inst.entity_type_id);
    auto interface_inst_id = GetLocalConstantId(inst.interface_type_id);
    if (HasNewWork()) {
      return Retry();
    }

    return ResolveAs<SemIR::AssociatedEntityType>(
        {.type_id = SemIR::TypeId::TypeType,
         .interface_type_id =
             context_.GetTypeIdForTypeConstant(interface_inst_id),
         .entity_type_id =
             context_.GetTypeIdForTypeConstant(entity_type_const_id)});
  }

  auto TryResolveTypedInst(SemIR::BaseDecl inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    auto type_const_id = GetLocalConstantId(inst.type_id);
    auto base_type_const_id = GetLocalConstantId(inst.base_type_id);
    if (HasNewWork()) {
      return Retry();
    }

    // Import the instruction in order to update contained base_type_id and
    // track the import location.
    auto inst_id = context_.AddInstInNoBlock(
        context_.MakeImportedLocAndInst<SemIR::BaseDecl>(
            AddImportIRInst(import_inst_id),
            {.type_id = context_.GetTypeIdForTypeConstant(type_const_id),
             .base_type_id =
                 context_.GetTypeIdForTypeConstant(base_type_const_id),
             .index = inst.index}));
    return ResolveAsConstant(context_.constant_values().Get(inst_id));
  }

  auto TryResolveTypedInst(SemIR::BindAlias inst) -> ResolveResult {
    auto value_id = GetLocalConstantId(inst.value_id);
    return RetryOrResolveAsConstant(value_id);
  }

  auto TryResolveTypedInst(SemIR::BindSymbolicName inst) -> ResolveResult {
    auto type_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork()) {
      return Retry();
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
  auto MakeIncompleteClass(const SemIR::Class& import_class,
                           SemIR::SpecificId enclosing_specific_id)
      -> std::pair<SemIR::ClassId, SemIR::ConstantId> {
    SemIR::ClassDecl class_decl = {.type_id = SemIR::TypeId::TypeType,
                                   .class_id = SemIR::ClassId::Invalid,
                                   .decl_block_id = SemIR::InstBlockId::Empty};
    auto class_decl_id =
        context_.AddPlaceholderInstInNoBlock(context_.MakeImportedLocAndInst(
            AddImportIRInst(import_class.latest_decl_id()), class_decl));
    // Regardless of whether ClassDecl is a complete type, we first need an
    // incomplete type so that any references have something to point at.
    class_decl.class_id = context_.classes().Add(
        {GetIncompleteLocalEntityBase(class_decl_id, import_class),
         {.self_type_id = SemIR::TypeId::Invalid,
          .inheritance_kind = import_class.inheritance_kind}});

    if (import_class.has_parameters()) {
      class_decl.type_id = context_.GetGenericClassType(class_decl.class_id,
                                                        enclosing_specific_id);
    }

    // Write the class ID into the ClassDecl.
    context_.ReplaceInstBeforeConstantUse(class_decl_id, class_decl);
    auto self_const_id = context_.constant_values().Get(class_decl_id);
    return {class_decl.class_id, self_const_id};
  }

  // Fills out the class definition for an incomplete class.
  auto AddClassDefinition(const SemIR::Class& import_class,
                          SemIR::Class& new_class,
                          SemIR::InstId complete_type_witness_id,
                          SemIR::InstId base_id) -> void {
    new_class.definition_id = new_class.first_owning_decl_id;

    new_class.complete_type_witness_id = complete_type_witness_id;

    new_class.scope_id = context_.name_scopes().Add(
        new_class.first_owning_decl_id, SemIR::NameId::Invalid,
        new_class.parent_scope_id);
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
      auto import_specific_id = SemIR::SpecificId::Invalid;
      if (auto import_generic_class_type =
              import_ir_.types().TryGetAs<SemIR::GenericClassType>(
                  inst.type_id)) {
        import_specific_id = import_generic_class_type->enclosing_specific_id;
      }
      auto specific_data = GetLocalSpecificData(import_specific_id);
      if (HasNewWork()) {
        // This is the end of the first phase. Don't make a new class yet if
        // we already have new work.
        return Retry();
      }

      // On the second phase, create a forward declaration of the class for any
      // recursive references.
      auto enclosing_specific_id =
          GetOrAddLocalSpecific(import_specific_id, specific_data);
      std::tie(class_id, class_const_id) =
          MakeIncompleteClass(import_class, enclosing_specific_id);
    } else {
      // On the third phase, compute the class ID from the constant
      // value of the declaration.
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
    auto parent_scope_id = GetLocalNameScopeId(import_class.parent_scope_id);
    LoadLocalParamConstantIds(import_class.implicit_param_refs_id);
    LoadLocalParamConstantIds(import_class.param_refs_id);
    auto generic_data = GetLocalGenericData(import_class.generic_id);
    auto self_const_id = GetLocalConstantId(import_class.self_type_id);
    auto complete_type_witness_id =
        import_class.complete_type_witness_id.is_valid()
            ? GetLocalConstantInstId(import_class.complete_type_witness_id)
            : SemIR::InstId::Invalid;
    auto base_id = import_class.base_id.is_valid()
                       ? GetLocalConstantInstId(import_class.base_id)
                       : SemIR::InstId::Invalid;

    if (HasNewWork()) {
      return Retry(class_const_id);
    }

    auto& new_class = context_.classes().Get(class_id);
    new_class.parent_scope_id = parent_scope_id;
    new_class.implicit_param_refs_id =
        GetLocalParamRefsId(import_class.implicit_param_refs_id);
    new_class.param_refs_id = GetLocalParamRefsId(import_class.param_refs_id);
    SetGenericData(import_class.generic_id, new_class.generic_id, generic_data);
    new_class.self_type_id = context_.GetTypeIdForTypeConstant(self_const_id);

    if (import_class.is_defined()) {
      AddClassDefinition(import_class, new_class, complete_type_witness_id,
                         base_id);
    }

    return ResolveAsConstant(class_const_id);
  }

  auto TryResolveTypedInst(SemIR::ClassType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id = GetLocalConstantId(
        import_ir_.classes().Get(inst.class_id).first_owning_decl_id);
    auto specific_data = GetLocalSpecificData(inst.specific_id);
    if (HasNewWork()) {
      return Retry();
    }

    // Find the corresponding class type. For a non-generic class, this is the
    // type of the class declaration. For a generic class, build a class type
    // referencing this specialization of the generic class.
    auto class_const_inst = context_.insts().Get(
        context_.constant_values().GetInstId(class_const_id));
    if (class_const_inst.Is<SemIR::ClassType>()) {
      return ResolveAsConstant(class_const_id);
    } else {
      auto generic_class_type = context_.types().GetAs<SemIR::GenericClassType>(
          class_const_inst.type_id());
      auto specific_id = GetOrAddLocalSpecific(inst.specific_id, specific_data);
      return ResolveAs<SemIR::ClassType>(
          {.type_id = SemIR::TypeId::TypeType,
           .class_id = generic_class_type.class_id,
           .specific_id = specific_id});
    }
  }

  auto TryResolveTypedInst(SemIR::CompleteTypeWitness inst) -> ResolveResult {
    CARBON_CHECK(import_ir_.types().GetInstId(inst.type_id) ==
                 SemIR::InstId::BuiltinWitnessType);
    auto object_repr_const_id = GetLocalConstantId(inst.object_repr_id);
    if (HasNewWork()) {
      return Retry();
    }
    auto object_repr_id =
        context_.GetTypeIdForTypeConstant(object_repr_const_id);
    return ResolveAs<SemIR::CompleteTypeWitness>(
        {.type_id =
             context_.GetBuiltinType(SemIR::BuiltinInstKind::WitnessType),
         .object_repr_id = object_repr_id});
  }

  auto TryResolveTypedInst(SemIR::ConstType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto inner_const_id = GetLocalConstantId(inst.inner_id);
    if (HasNewWork()) {
      return Retry();
    }
    auto inner_type_id = context_.GetTypeIdForTypeConstant(inner_const_id);
    return ResolveAs<SemIR::ConstType>(
        {.type_id = SemIR::TypeId::TypeType, .inner_id = inner_type_id});
  }

  auto TryResolveTypedInst(SemIR::ExportDecl inst) -> ResolveResult {
    auto value_id = GetLocalConstantId(inst.value_id);
    return RetryOrResolveAsConstant(value_id);
  }

  auto TryResolveTypedInst(SemIR::FieldDecl inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    auto const_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork()) {
      return Retry();
    }
    auto inst_id = context_.AddInstInNoBlock(
        context_.MakeImportedLocAndInst<SemIR::FieldDecl>(
            AddImportIRInst(import_inst_id),
            {.type_id = context_.GetTypeIdForTypeConstant(const_id),
             .name_id = GetLocalNameId(inst.name_id),
             .index = inst.index}));
    return {.const_id = context_.constant_values().Get(inst_id)};
  }

  // Make a declaration of a function. This is done as a separate step from
  // importing the function declaration in order to resolve cycles.
  auto MakeFunctionDecl(const SemIR::Function& import_function,
                        SemIR::SpecificId specific_id)
      -> std::pair<SemIR::FunctionId, SemIR::ConstantId> {
    SemIR::FunctionDecl function_decl = {
        .type_id = SemIR::TypeId::Invalid,
        .function_id = SemIR::FunctionId::Invalid,
        .decl_block_id = SemIR::InstBlockId::Empty};
    auto function_decl_id =
        context_.AddPlaceholderInstInNoBlock(context_.MakeImportedLocAndInst(
            AddImportIRInst(import_function.first_decl_id()), function_decl));

    // Start with an incomplete function.
    function_decl.function_id = context_.functions().Add(
        {GetIncompleteLocalEntityBase(function_decl_id, import_function),
         {.return_storage_id = SemIR::InstId::Invalid,
          .builtin_function_kind = import_function.builtin_function_kind}});

    function_decl.type_id =
        context_.GetFunctionType(function_decl.function_id, specific_id);

    // Write the function ID and type into the FunctionDecl.
    context_.ReplaceInstBeforeConstantUse(function_decl_id, function_decl);
    return {function_decl.function_id,
            context_.constant_values().Get(function_decl_id)};
  }

  auto TryResolveTypedInst(SemIR::FunctionDecl inst,
                           SemIR::ConstantId function_const_id)
      -> ResolveResult {
    const auto& import_function = import_ir_.functions().Get(inst.function_id);

    SemIR::FunctionId function_id = SemIR::FunctionId::Invalid;
    if (!function_const_id.is_valid()) {
      auto import_specific_id = import_ir_.types()
                                    .GetAs<SemIR::FunctionType>(inst.type_id)
                                    .specific_id;
      auto specific_data = GetLocalSpecificData(import_specific_id);
      if (HasNewWork()) {
        // This is the end of the first phase. Don't make a new function yet if
        // we already have new work.
        return Retry();
      }

      // On the second phase, create a forward declaration of the interface.
      auto specific_id =
          GetOrAddLocalSpecific(import_specific_id, specific_data);
      std::tie(function_id, function_const_id) =
          MakeFunctionDecl(import_function, specific_id);
    } else {
      // On the third phase, compute the function ID from the constant value of
      // the declaration.
      auto function_const_inst = context_.insts().Get(
          context_.constant_values().GetInstId(function_const_id));
      auto function_type = context_.types().GetAs<SemIR::FunctionType>(
          function_const_inst.type_id());
      function_id = function_type.function_id;
    }

    auto return_type_const_id = SemIR::ConstantId::Invalid;
    if (import_function.return_storage_id.is_valid()) {
      return_type_const_id = GetLocalConstantId(
          import_ir_.insts().Get(import_function.return_storage_id).type_id());
    }
    auto parent_scope_id = GetLocalNameScopeId(import_function.parent_scope_id);
    LoadLocalParamConstantIds(import_function.implicit_param_refs_id);
    LoadLocalParamConstantIds(import_function.param_refs_id);
    auto generic_data = GetLocalGenericData(import_function.generic_id);

    if (HasNewWork()) {
      return Retry(function_const_id);
    }

    // Add the function declaration.
    auto& new_function = context_.functions().Get(function_id);
    new_function.parent_scope_id = parent_scope_id;
    new_function.implicit_param_refs_id =
        GetLocalParamRefsId(import_function.implicit_param_refs_id);
    new_function.param_refs_id =
        GetLocalParamRefsId(import_function.param_refs_id);
    SetGenericData(import_function.generic_id, new_function.generic_id,
                   generic_data);

    if (import_function.return_storage_id.is_valid()) {
      // Recreate the return slot from scratch.
      // TODO: Once we import function definitions, we'll need to make sure we
      // use the same return storage variable in the declaration and definition.
      new_function.return_storage_id =
          context_.AddInstInNoBlock<SemIR::VarStorage>(
              AddImportIRInst(import_function.return_storage_id),
              {.type_id =
                   context_.GetTypeIdForTypeConstant(return_type_const_id),
               .name_id = SemIR::NameId::ReturnSlot});
    }

    if (import_function.definition_id.is_valid()) {
      new_function.definition_id = new_function.first_owning_decl_id;
    }

    return ResolveAsConstant(function_const_id);
  }

  auto TryResolveTypedInst(SemIR::FunctionType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto fn_val_id = GetLocalConstantInstId(
        import_ir_.functions().Get(inst.function_id).first_decl_id());
    auto specific_data = GetLocalSpecificData(inst.specific_id);
    if (HasNewWork()) {
      return Retry();
    }
    auto fn_type_id = context_.insts().Get(fn_val_id).type_id();
    return ResolveAs<SemIR::FunctionType>(
        {.type_id = SemIR::TypeId::TypeType,
         .function_id = context_.types()
                            .GetAs<SemIR::FunctionType>(fn_type_id)
                            .function_id,
         .specific_id =
             GetOrAddLocalSpecific(inst.specific_id, specific_data)});
  }

  auto TryResolveTypedInst(SemIR::GenericClassType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_val_id = GetLocalConstantInstId(
        import_ir_.classes().Get(inst.class_id).first_owning_decl_id);
    if (HasNewWork()) {
      return Retry();
    }
    auto class_val = context_.insts().Get(class_val_id);
    CARBON_CHECK(
        context_.types().Is<SemIR::GenericClassType>(class_val.type_id()));
    return ResolveAsConstant(
        context_.types().GetConstantId(class_val.type_id()));
  }

  auto TryResolveTypedInst(SemIR::GenericInterfaceType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto interface_val_id = GetLocalConstantInstId(
        import_ir_.interfaces().Get(inst.interface_id).first_owning_decl_id);
    if (HasNewWork()) {
      return Retry();
    }
    auto interface_val = context_.insts().Get(interface_val_id);
    CARBON_CHECK(context_.types().Is<SemIR::GenericInterfaceType>(
        interface_val.type_id()));
    return ResolveAsConstant(
        context_.types().GetConstantId(interface_val.type_id()));
  }

  // Make a declaration of an impl. This is done as a separate step from
  // importing the impl definition in order to resolve cycles.
  auto MakeImplDeclaration(const SemIR::Impl& import_impl)
      -> std::pair<SemIR::ImplId, SemIR::ConstantId> {
    SemIR::ImplDecl impl_decl = {.impl_id = SemIR::ImplId::Invalid,
                                 .decl_block_id = SemIR::InstBlockId::Empty};
    auto impl_decl_id =
        context_.AddPlaceholderInstInNoBlock(context_.MakeImportedLocAndInst(
            AddImportIRInst(import_impl.latest_decl_id()), impl_decl));
    impl_decl.impl_id = context_.impls().Add(
        {GetIncompleteLocalEntityBase(impl_decl_id, import_impl),
         {.self_id = SemIR::InstId::Invalid,
          .constraint_id = SemIR::InstId::Invalid,
          .witness_id = SemIR::InstId::Invalid}});

    // Write the impl ID into the ImplDecl.
    context_.ReplaceInstBeforeConstantUse(impl_decl_id, impl_decl);
    return {impl_decl.impl_id, context_.constant_values().Get(impl_decl_id)};
  }

  // Imports the definition of an impl.
  auto AddImplDefinition(const SemIR::Impl& import_impl, SemIR::Impl& new_impl,
                         SemIR::InstId witness_id) -> void {
    new_impl.definition_id = new_impl.first_owning_decl_id;

    new_impl.witness_id = witness_id;

    // Import the definition scope, if we might need it. Name lookup is never
    // performed into this scope by a user of the impl, so this is only
    // necessary in the same library that defined the impl, in order to support
    // defining members of the impl out of line in the impl file when the impl
    // is defined in the API file.
    if (import_ir_id_ == SemIR::ImportIRId::ApiForImpl) {
      new_impl.scope_id = context_.name_scopes().Add(
          new_impl.first_owning_decl_id, SemIR::NameId::Invalid,
          new_impl.parent_scope_id);
      auto& new_scope = context_.name_scopes().Get(new_impl.scope_id);
      const auto& import_scope =
          import_ir_.name_scopes().Get(import_impl.scope_id);

      // Push a block so that we can add scoped instructions to it.
      context_.inst_block_stack().Push();
      AddNameScopeImportRefs(import_scope, new_scope);
      new_impl.body_block_id = context_.inst_block_stack().Pop();
    }
  }

  auto TryResolveTypedInst(SemIR::ImplDecl inst,
                           SemIR::ConstantId impl_const_id) -> ResolveResult {
    // TODO: This duplicates a lot of the handling of interfaces, classes, and
    // functions. Factor out the commonality.
    const auto& import_impl = import_ir_.impls().Get(inst.impl_id);

    SemIR::ImplId impl_id = SemIR::ImplId::Invalid;
    if (!impl_const_id.is_valid()) {
      if (HasNewWork()) {
        // This is the end of the first phase. Don't make a new impl yet if we
        // already have new work.
        return Retry();
      }

      // On the second phase, create a forward declaration of the impl for any
      // recursive references.
      std::tie(impl_id, impl_const_id) = MakeImplDeclaration(import_impl);
    } else {
      // On the third phase, compute the impl ID from the "constant value" of
      // the declaration, which is a reference to the created ImplDecl.
      auto impl_const_inst = context_.insts().GetAs<SemIR::ImplDecl>(
          context_.constant_values().GetInstId(impl_const_id));
      impl_id = impl_const_inst.impl_id;
    }

    // Load constants for the definition.
    auto parent_scope_id = GetLocalNameScopeId(import_impl.parent_scope_id);
    LoadLocalParamConstantIds(import_impl.implicit_param_refs_id);
    auto generic_data = GetLocalGenericData(import_impl.generic_id);
    auto self_const_id = GetLocalConstantId(
        import_ir_.constant_values().Get(import_impl.self_id));
    auto constraint_const_id = GetLocalConstantId(
        import_ir_.constant_values().Get(import_impl.constraint_id));

    if (HasNewWork()) {
      return Retry(impl_const_id);
    }

    auto& new_impl = context_.impls().Get(impl_id);
    new_impl.parent_scope_id = parent_scope_id;
    new_impl.implicit_param_refs_id =
        GetLocalParamRefsId(import_impl.implicit_param_refs_id);
    CARBON_CHECK(!import_impl.param_refs_id.is_valid() &&
                 !new_impl.param_refs_id.is_valid());
    SetGenericData(import_impl.generic_id, new_impl.generic_id, generic_data);

    // Create instructions for self and constraint to hold the symbolic constant
    // value for a generic impl.
    new_impl.self_id = AddLoadedImportRef(
        context_, {.ir_id = import_ir_id_, .inst_id = import_impl.self_id},
        SemIR::TypeId::TypeType, self_const_id);
    new_impl.constraint_id = AddLoadedImportRef(
        context_,
        {.ir_id = import_ir_id_, .inst_id = import_impl.constraint_id},
        SemIR::TypeId::TypeType, constraint_const_id);

    if (import_impl.is_defined()) {
      auto witness_id = AddImportRef(
          context_,
          {.ir_id = import_ir_id_, .inst_id = import_impl.witness_id});
      AddImplDefinition(import_impl, new_impl, witness_id);
    }

    // If the `impl` is declared in the API file corresponding to the current
    // file, add this to impl lookup so that it can be found by redeclarations
    // in the current file.
    if (import_ir_id_ == SemIR::ImportIRId::ApiForImpl) {
      context_.impls().GetOrAddLookupBucket(new_impl).push_back(impl_id);
    }

    return ResolveAsConstant(impl_const_id);
  }

  auto TryResolveTypedInst(SemIR::ImportRefLoaded /*inst*/,
                           SemIR::InstId inst_id) -> ResolveResult {
    // Return the constant for the instruction of the imported constant.
    auto constant_id = import_ir_.constant_values().Get(inst_id);
    if (!constant_id.is_valid()) {
      return ResolveAsConstant(SemIR::ConstantId::Error);
    }
    if (!constant_id.is_constant()) {
      context_.TODO(inst_id,
                    "Non-constant ImportRefLoaded (comes up with var)");
      return ResolveAsConstant(SemIR::ConstantId::Error);
    }

    auto new_constant_id =
        GetLocalConstantId(import_ir_.constant_values().GetInstId(constant_id));
    return RetryOrResolveAsConstant(new_constant_id);
  }

  // Make a declaration of an interface. This is done as a separate step from
  // importing the interface definition in order to resolve cycles.
  auto MakeInterfaceDecl(const SemIR::Interface& import_interface,
                         SemIR::SpecificId enclosing_specific_id)
      -> std::pair<SemIR::InterfaceId, SemIR::ConstantId> {
    SemIR::InterfaceDecl interface_decl = {
        .type_id = SemIR::TypeId::TypeType,
        .interface_id = SemIR::InterfaceId::Invalid,
        .decl_block_id = SemIR::InstBlockId::Empty};
    auto interface_decl_id =
        context_.AddPlaceholderInstInNoBlock(context_.MakeImportedLocAndInst(
            AddImportIRInst(import_interface.first_owning_decl_id),
            interface_decl));

    // Start with an incomplete interface.
    interface_decl.interface_id = context_.interfaces().Add(
        {GetIncompleteLocalEntityBase(interface_decl_id, import_interface),
         {}});

    if (import_interface.has_parameters()) {
      interface_decl.type_id = context_.GetGenericInterfaceType(
          interface_decl.interface_id, enclosing_specific_id);
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
        new_interface.first_owning_decl_id, SemIR::NameId::Invalid,
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

    CARBON_CHECK(import_scope.extended_scopes.empty(),
                 "Interfaces don't currently have extended scopes to support.");
  }

  auto TryResolveTypedInst(SemIR::InterfaceDecl inst,
                           SemIR::ConstantId interface_const_id)
      -> ResolveResult {
    const auto& import_interface =
        import_ir_.interfaces().Get(inst.interface_id);

    SemIR::InterfaceId interface_id = SemIR::InterfaceId::Invalid;
    if (!interface_const_id.is_valid()) {
      auto import_specific_id = SemIR::SpecificId::Invalid;
      if (auto import_generic_interface_type =
              import_ir_.types().TryGetAs<SemIR::GenericInterfaceType>(
                  inst.type_id)) {
        import_specific_id =
            import_generic_interface_type->enclosing_specific_id;
      }
      auto specific_data = GetLocalSpecificData(import_specific_id);
      if (HasNewWork()) {
        // This is the end of the first phase. Don't make a new interface yet if
        // we already have new work.
        return Retry();
      }

      // On the second phase, create a forward declaration of the interface.
      auto enclosing_specific_id =
          GetOrAddLocalSpecific(import_specific_id, specific_data);
      std::tie(interface_id, interface_const_id) =
          MakeInterfaceDecl(import_interface, enclosing_specific_id);
    } else {
      // On the third phase, compute the interface ID from the constant value of
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

    auto parent_scope_id =
        GetLocalNameScopeId(import_interface.parent_scope_id);
    LoadLocalParamConstantIds(import_interface.implicit_param_refs_id);
    LoadLocalParamConstantIds(import_interface.param_refs_id);
    auto generic_data = GetLocalGenericData(import_interface.generic_id);

    std::optional<SemIR::InstId> self_param_id;
    if (import_interface.is_defined()) {
      self_param_id = GetLocalConstantInstId(import_interface.self_param_id);
    }

    if (HasNewWork()) {
      return Retry(interface_const_id);
    }

    auto& new_interface = context_.interfaces().Get(interface_id);
    new_interface.parent_scope_id = parent_scope_id;
    new_interface.implicit_param_refs_id =
        GetLocalParamRefsId(import_interface.implicit_param_refs_id);
    new_interface.param_refs_id =
        GetLocalParamRefsId(import_interface.param_refs_id);
    SetGenericData(import_interface.generic_id, new_interface.generic_id,
                   generic_data);

    if (import_interface.is_defined()) {
      CARBON_CHECK(self_param_id);
      AddInterfaceDefinition(import_interface, new_interface, *self_param_id);
    }
    return ResolveAsConstant(interface_const_id);
  }

  auto TryResolveTypedInst(SemIR::InterfaceType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto interface_const_id = GetLocalConstantId(
        import_ir_.interfaces().Get(inst.interface_id).first_owning_decl_id);
    auto specific_data = GetLocalSpecificData(inst.specific_id);
    if (HasNewWork()) {
      return Retry();
    }

    // Find the corresponding interface type. For a non-generic interface, this
    // is the type of the interface declaration. For a generic interface, build
    // a interface type referencing this specialization of the generic
    // interface.
    auto interface_const_inst = context_.insts().Get(
        context_.constant_values().GetInstId(interface_const_id));
    if (interface_const_inst.Is<SemIR::InterfaceType>()) {
      return ResolveAsConstant(interface_const_id);
    } else {
      auto generic_interface_type =
          context_.types().GetAs<SemIR::GenericInterfaceType>(
              interface_const_inst.type_id());
      auto specific_id = GetOrAddLocalSpecific(inst.specific_id, specific_data);
      return ResolveAs<SemIR::InterfaceType>(
          {.type_id = SemIR::TypeId::TypeType,
           .interface_id = generic_interface_type.interface_id,
           .specific_id = specific_id});
    }
  }

  auto TryResolveTypedInst(SemIR::InterfaceWitness inst) -> ResolveResult {
    auto elements = GetLocalInstBlockContents(inst.elements_id);
    if (HasNewWork()) {
      return Retry();
    }

    auto elements_id = GetLocalCanonicalInstBlockId(inst.elements_id, elements);
    return ResolveAs<SemIR::InterfaceWitness>(
        {.type_id =
             context_.GetBuiltinType(SemIR::BuiltinInstKind::WitnessType),
         .elements_id = elements_id});
  }

  auto TryResolveTypedInst(SemIR::IntLiteral inst) -> ResolveResult {
    auto type_id = GetLocalConstantId(inst.type_id);
    if (HasNewWork()) {
      return Retry();
    }

    return ResolveAs<SemIR::IntLiteral>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .int_id = context_.ints().Add(import_ir_.ints().Get(inst.int_id))});
  }

  auto TryResolveTypedInst(SemIR::PointerType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto pointee_const_id = GetLocalConstantId(inst.pointee_id);
    if (HasNewWork()) {
      return Retry();
    }

    auto pointee_type_id = context_.GetTypeIdForTypeConstant(pointee_const_id);
    return ResolveAs<SemIR::PointerType>(
        {.type_id = SemIR::TypeId::TypeType, .pointee_id = pointee_type_id});
  }

  auto TryResolveTypedInst(SemIR::StructType inst, SemIR::InstId import_inst_id)
      -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto orig_fields = import_ir_.inst_blocks().Get(inst.fields_id);
    llvm::SmallVector<SemIR::ConstantId> field_const_ids;
    field_const_ids.reserve(orig_fields.size());
    for (auto field_id : orig_fields) {
      auto field = import_ir_.insts().GetAs<SemIR::StructTypeField>(field_id);
      field_const_ids.push_back(GetLocalConstantId(field.field_type_id));
    }
    if (HasNewWork()) {
      return Retry();
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
    auto type_id = GetLocalConstantId(inst.type_id);
    auto elems = GetLocalInstBlockContents(inst.elements_id);
    if (HasNewWork()) {
      return Retry();
    }

    return ResolveAs<SemIR::StructValue>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .elements_id = GetLocalCanonicalInstBlockId(inst.elements_id, elems)});
  }

  auto TryResolveTypedInst(SemIR::TupleType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

    auto orig_elem_type_ids = import_ir_.type_blocks().Get(inst.elements_id);
    llvm::SmallVector<SemIR::ConstantId> elem_const_ids;
    elem_const_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_type_id : orig_elem_type_ids) {
      elem_const_ids.push_back(GetLocalConstantId(elem_type_id));
    }
    if (HasNewWork()) {
      return Retry();
    }

    // Prepare a vector of the tuple types for GetTupleType.
    llvm::SmallVector<SemIR::TypeId> elem_type_ids;
    elem_type_ids.reserve(orig_elem_type_ids.size());
    for (auto elem_const_id : elem_const_ids) {
      elem_type_ids.push_back(context_.GetTypeIdForTypeConstant(elem_const_id));
    }

    return ResolveAsConstant(
        context_.types().GetConstantId(context_.GetTupleType(elem_type_ids)));
  }

  auto TryResolveTypedInst(SemIR::TupleValue inst) -> ResolveResult {
    auto type_id = GetLocalConstantId(inst.type_id);
    auto elems = GetLocalInstBlockContents(inst.elements_id);
    if (HasNewWork()) {
      return Retry();
    }

    return ResolveAs<SemIR::TupleValue>(
        {.type_id = context_.GetTypeIdForTypeConstant(type_id),
         .elements_id = GetLocalCanonicalInstBlockId(inst.elements_id, elems)});
  }

  auto TryResolveTypedInst(SemIR::UnboundElementType inst) -> ResolveResult {
    CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
    auto class_const_id = GetLocalConstantId(inst.class_type_id);
    auto elem_const_id = GetLocalConstantId(inst.element_type_id);
    if (HasNewWork()) {
      return Retry();
    }

    return ResolveAs<SemIR::UnboundElementType>(
        {.type_id = SemIR::TypeId::TypeType,
         .class_type_id = context_.GetTypeIdForTypeConstant(class_const_id),
         .element_type_id = context_.GetTypeIdForTypeConstant(elem_const_id)});
  }

  // Perform any work that we deferred until the end of the main Resolve loop.
  auto PerformPendingWork() -> void {
    // Note that the individual Finish steps can add new pending work, so keep
    // going until we have no more work to do.
    while (!pending_generics_.empty() || !pending_specifics_.empty()) {
      // Process generics in the order that we added them because a later
      // generic might refer to an earlier one, and the calls to
      // RebuildGenericEvalBlock assume that the reachable SemIR is in a valid
      // state.
      // TODO: Import the generic eval block rather than calling
      // RebuildGenericEvalBlock to rebuild it so that order doesn't matter.
      // NOLINTNEXTLINE(modernize-loop-convert)
      for (size_t i = 0; i != pending_generics_.size(); ++i) {
        FinishPendingGeneric(pending_generics_[i]);
      }
      pending_generics_.clear();

      while (!pending_specifics_.empty()) {
        FinishPendingSpecific(pending_specifics_.pop_back_val());
      }
    }
  }

  // Resolves and returns the local contents for an imported instruction block
  // of constant instructions.
  auto ResolveLocalInstBlockContents(SemIR::InstBlockId import_block_id)
      -> llvm::SmallVector<SemIR::InstId> {
    auto import_block = import_ir_.inst_blocks().Get(import_block_id);

    llvm::SmallVector<SemIR::InstId> inst_ids;
    inst_ids.reserve(import_block.size());
    for (auto import_inst_id : import_block) {
      inst_ids.push_back(
          context_.constant_values().GetInstId(ResolveOneInst(import_inst_id)));
    }
    return inst_ids;
  }

  // Resolves and returns a local eval block for a region of an imported
  // generic.
  auto ResolveLocalEvalBlock(const SemIR::Generic& import_generic,
                             SemIR::GenericId generic_id,
                             SemIR::GenericInstIndex::Region region)
      -> SemIR::InstBlockId {
    auto import_block_id = import_generic.GetEvalBlock(region);
    if (!import_block_id.is_valid()) {
      return SemIR::InstBlockId::Invalid;
    }

    auto inst_ids = ResolveLocalInstBlockContents(import_block_id);
    return RebuildGenericEvalBlock(context_, generic_id, region, inst_ids);
  }

  // Fills in the remaining information in a partially-imported generic.
  auto FinishPendingGeneric(PendingGeneric pending) -> void {
    const auto& import_generic = import_ir_.generics().Get(pending.import_id);

    // Don't store the local generic between calls: the generics list can be
    // reallocated by ResolveLocalEvalBlock importing more specifics.

    auto decl_block_id =
        ResolveLocalEvalBlock(import_generic, pending.local_id,
                              SemIR::GenericInstIndex::Region::Declaration);
    context_.generics().Get(pending.local_id).decl_block_id = decl_block_id;

    auto self_specific_id = MakeSelfSpecific(context_, pending.local_id);
    context_.generics().Get(pending.local_id).self_specific_id =
        self_specific_id;
    pending_specifics_.push_back({.import_id = import_generic.self_specific_id,
                                  .local_id = self_specific_id});

    auto definition_block_id =
        ResolveLocalEvalBlock(import_generic, pending.local_id,
                              SemIR::GenericInstIndex::Region::Definition);
    context_.generics().Get(pending.local_id).definition_block_id =
        definition_block_id;
  }

  // Resolves and returns a local inst block of constant instructions
  // corresponding to an imported inst block.
  auto ResolveLocalInstBlock(SemIR::InstBlockId import_block_id)
      -> SemIR::InstBlockId {
    if (!import_block_id.is_valid()) {
      return SemIR::InstBlockId::Invalid;
    }

    auto inst_ids = ResolveLocalInstBlockContents(import_block_id);
    return context_.inst_blocks().Add(inst_ids);
  }

  // Fills in the remaining information in a partially-imported specific.
  auto FinishPendingSpecific(PendingSpecific pending) -> void {
    const auto& import_specific = import_ir_.specifics().Get(pending.import_id);

    // Don't store the local specific between calls: the specifics list can be
    // reallocated by ResolveLocalInstBlock importing more specifics.

    if (!context_.specifics().Get(pending.local_id).decl_block_id.is_valid()) {
      auto decl_block_id = ResolveLocalInstBlock(import_specific.decl_block_id);
      context_.specifics().Get(pending.local_id).decl_block_id = decl_block_id;
    }

    if (!context_.specifics()
             .Get(pending.local_id)
             .definition_block_id.is_valid() &&
        import_specific.definition_block_id.is_valid()) {
      auto definition_block_id =
          ResolveLocalInstBlock(import_specific.definition_block_id);
      context_.specifics().Get(pending.local_id).definition_block_id =
          definition_block_id;
    }
  }

  auto import_ir_constant_values() -> SemIR::ConstantValueStore& {
    return context_.import_ir_constant_values()[import_ir_id_.index];
  }

  Context& context_;
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File& import_ir_;
  llvm::SmallVector<Work> work_stack_;
  // The size of work_stack_ at the start of resolving the current instruction.
  size_t initial_work_ = 0;
  // Generics that we have partially imported but not yet finished importing.
  llvm::SmallVector<PendingGeneric> pending_generics_;
  // Specifics that we have partially imported but not yet finished importing.
  llvm::SmallVector<PendingSpecific> pending_specifics_;
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

// TODO: This doesn't belong in this file. Consider moving the import resolver
// and this file elsewhere.
auto ImportImpls(Context& context) -> void {
  for (auto [import_index, import_ir] :
       llvm::enumerate(context.import_irs().array_ref())) {
    if (!import_ir.sem_ir) {
      continue;
    }

    SemIR::ImportIRId import_ir_id(import_index);
    for (auto impl_index : llvm::seq(import_ir.sem_ir->impls().size())) {
      SemIR::ImplId impl_id(impl_index);

      // Resolve the imported impl to a local impl ID.
      ImportRefResolver resolver(context, import_ir_id);
      resolver.Resolve(import_ir.sem_ir->impls().Get(impl_id).first_decl_id());
    }
  }
}

}  // namespace Carbon::Check
