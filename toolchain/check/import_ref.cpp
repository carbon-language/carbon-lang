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
static auto AddLoadedImportRef(Context& context, SemIR::TypeId type_id,
                               SemIR::ImportIRInst import_ir_inst,
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

namespace {
class ImportRefResolver;

// The result of attempting to resolve an imported instruction to a constant.
struct ResolveResult {
  // The new constant value, if known.
  SemIR::ConstantId const_id;
  // Whether resolution has been attempted once and needs to be retried.
  bool retry = false;

  // Produces a resolve result that tries resolving this instruction again. If
  // `const_id` is specified, then this is the end of the second phase, and the
  // constant value will be passed to the next resolution attempt. Otherwise,
  // this is the end of the first phase.
  static auto Retry(SemIR::ConstantId const_id = SemIR::ConstantId::Invalid)
      -> ResolveResult {
    return {.const_id = const_id, .retry = true};
  }

  // Produces a resolve result that provides the given constant value. Requires
  // that there is no new work.
  static auto Done(SemIR::ConstantId const_id) -> ResolveResult {
    return {.const_id = const_id};
  }
};
}  // namespace

static auto TryResolveInst(ImportRefResolver& resolver, SemIR::InstId inst_id,
                           SemIR::ConstantId const_id) -> ResolveResult;

namespace {
// A context within which we are performing an import. Tracks information about
// the source and destination. This provides a restricted interface compared to
// ImportResolver: in particular, it does not have access to a work list.
// Therefore code that accepts an ImportContext is unable to enqueue new work.
class ImportContext {
 public:
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

  explicit ImportContext(Context& context, SemIR::ImportIRId import_ir_id)
      : context_(context),
        import_ir_id_(import_ir_id),
        import_ir_(*context_.import_irs().Get(import_ir_id).sem_ir) {}

  // Returns the file we are importing from.
  auto import_ir() -> const SemIR::File& { return import_ir_; }

  // Accessors into value stores of the file we are importing from.
  auto import_classes() -> decltype(auto) { return import_ir().classes(); }
  auto import_constant_values() -> decltype(auto) {
    return import_ir().constant_values();
  }
  auto import_entity_names() -> decltype(auto) {
    return import_ir().entity_names();
  }
  auto import_facet_types() -> decltype(auto) {
    return import_ir().facet_types();
  }
  auto import_functions() -> decltype(auto) { return import_ir().functions(); }
  auto import_generics() -> decltype(auto) { return import_ir().generics(); }
  auto import_identifiers() -> decltype(auto) {
    return import_ir().identifiers();
  }
  auto import_impls() -> decltype(auto) { return import_ir().impls(); }
  auto import_inst_blocks() -> decltype(auto) {
    return import_ir().inst_blocks();
  }
  auto import_insts() -> decltype(auto) { return import_ir().insts(); }
  auto import_interfaces() -> decltype(auto) {
    return import_ir().interfaces();
  }
  auto import_ints() -> decltype(auto) { return import_ir().ints(); }
  auto import_name_scopes() -> decltype(auto) {
    return import_ir().name_scopes();
  }
  auto import_specifics() -> decltype(auto) { return import_ir().specifics(); }
  auto import_string_literal_values() -> decltype(auto) {
    return import_ir().string_literal_values();
  }
  auto import_struct_type_fields() -> decltype(auto) {
    return import_ir().struct_type_fields();
  }
  auto import_type_blocks() -> decltype(auto) {
    return import_ir().type_blocks();
  }
  auto import_types() -> decltype(auto) { return import_ir().types(); }

  // Returns the local file's import ID for the IR we are importing from.
  auto import_ir_id() -> SemIR::ImportIRId { return import_ir_id_; }

  // A value store for local constant values of imported instructions. This maps
  // from `InstId`s in the import IR to corresponding `ConstantId`s in the local
  // IR.
  auto local_constant_values_for_import_insts() -> decltype(auto) {
    return local_context().import_ir_constant_values()[import_ir_id_.index];
  }

  // Returns the file we are importing into.
  auto local_ir() -> SemIR::File& { return context_.sem_ir(); }

  // Returns the type-checking context we are importing into.
  auto local_context() -> Context& { return context_; }

  // Accessors into value stores of the file we are importing into.
  auto local_classes() -> decltype(auto) { return local_ir().classes(); }
  auto local_constant_values() -> decltype(auto) {
    return local_ir().constant_values();
  }
  auto local_entity_names() -> decltype(auto) {
    return local_ir().entity_names();
  }
  auto local_facet_types() -> decltype(auto) {
    return local_ir().facet_types();
  }
  auto local_functions() -> decltype(auto) { return local_ir().functions(); }
  auto local_generics() -> decltype(auto) { return local_ir().generics(); }
  auto local_identifiers() -> decltype(auto) {
    return local_ir().identifiers();
  }
  auto local_impls() -> decltype(auto) { return local_ir().impls(); }
  auto local_import_ir_insts() -> decltype(auto) {
    return local_ir().import_ir_insts();
  }
  auto local_inst_blocks() -> decltype(auto) {
    return local_ir().inst_blocks();
  }
  auto local_insts() -> decltype(auto) { return local_ir().insts(); }
  auto local_interfaces() -> decltype(auto) { return local_ir().interfaces(); }
  auto local_ints() -> decltype(auto) { return local_ir().ints(); }
  auto local_name_scopes() -> decltype(auto) {
    return local_ir().name_scopes();
  }
  auto local_specifics() -> decltype(auto) { return local_ir().specifics(); }
  auto local_string_literal_values() -> decltype(auto) {
    return local_ir().string_literal_values();
  }
  auto local_struct_type_fields() -> decltype(auto) {
    return local_ir().struct_type_fields();
  }
  auto local_types() -> decltype(auto) { return local_ir().types(); }

  // Add a generic that has been partially imported but needs to be finished.
  auto AddPendingGeneric(PendingGeneric pending) -> void {
    pending_generics_.push_back(pending);
  }

  // Add a specific that has been partially imported but needs to be finished.
  auto AddPendingSpecific(PendingSpecific pending) -> void {
    pending_specifics_.push_back(pending);
  }

 protected:
  Context& context_;
  SemIR::ImportIRId import_ir_id_;
  const SemIR::File& import_ir_;

  // TODO: The following members don't belong here. This pending work mechanism
  // can probably be removed entirely if we stop importing generic eval blocks
  // and instead evaluate them directly in the imported IR.

  // Generics that we have partially imported but not yet finished importing.
  llvm::SmallVector<PendingGeneric> pending_generics_;
  // Specifics that we have partially imported but not yet finished importing.
  llvm::SmallVector<PendingSpecific> pending_specifics_;
};

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
class ImportRefResolver : public ImportContext {
 public:
  explicit ImportRefResolver(Context& context, SemIR::ImportIRId import_ir_id)
      : ImportContext(context, import_ir_id) {}

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
          TryResolveInst(*this, work.inst_id, existing.const_id);
      CARBON_CHECK(!HasNewWork() || retry);

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
    auto constant_id = local_constant_values_for_import_insts().Get(inst_id);
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
    return Resolve(GetInstWithConstantValue(import_ir(), import_const_id));
  }

  // Wraps constant evaluation with logic to handle types.
  auto ResolveType(SemIR::TypeId import_type_id) -> SemIR::TypeId {
    if (!import_type_id.is_valid()) {
      return import_type_id;
    }

    auto import_type_const_id =
        import_ir().types().GetConstantId(import_type_id);
    CARBON_CHECK(import_type_const_id.is_valid());

    if (auto import_type_inst_id =
            import_ir().constant_values().GetInstId(import_type_const_id);
        import_type_inst_id.is_builtin()) {
      // Builtins don't require constant resolution; we can use them directly.
      return local_context().GetBuiltinType(
          import_type_inst_id.builtin_inst_kind());
    } else {
      return local_context().GetTypeIdForTypeConstant(
          ResolveConstant(import_type_id.AsConstantId()));
    }
  }

  // Returns true if new unresolved constants were found as part of this
  // `Resolve` step.
  auto HasNewWork() -> bool {
    CARBON_CHECK(initial_work_ <= work_stack_.size(),
                 "Work shouldn't decrease");
    return initial_work_ < work_stack_.size();
  }

  // Returns the ConstantId for an InstId. Adds unresolved constants to
  // work_stack_.
  auto GetLocalConstantValueOrPush(SemIR::InstId inst_id) -> SemIR::ConstantId {
    auto const_id = local_constant_values_for_import_insts().Get(inst_id);
    if (!const_id.is_valid()) {
      work_stack_.push_back({.inst_id = inst_id});
    }
    return const_id;
  }

 private:
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

  // Looks to see if an instruction has been resolved. If a constant is only
  // found indirectly, sets the constant for any indirect steps that don't
  // already have the constant. If a constant isn't found, returns the indirect
  // instructions so that they can have the resolved constant assigned later.
  auto FindResolvedConstId(SemIR::InstId inst_id) -> ResolvedConstId {
    ResolvedConstId result;

    if (auto existing_const_id =
            local_constant_values_for_import_insts().Get(inst_id);
        existing_const_id.is_valid()) {
      result.const_id = existing_const_id;
      return result;
    }

    const auto* cursor_ir = &import_ir();
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
      cursor_ir_id = local_context().GetImportIRId(*cursor_ir);
      if (!cursor_ir_id.is_valid()) {
        // TODO: Should we figure out a location to assign here?
        cursor_ir_id =
            AddImportIR(local_context(), {.decl_id = SemIR::InstId::Invalid,
                                          .is_export = false,
                                          .sem_ir = cursor_ir});
      }
      cursor_inst_id = ir_inst.inst_id;

      CARBON_CHECK(cursor_ir != prev_ir || cursor_inst_id != prev_inst_id,
                   "{0}", cursor_ir->insts().Get(cursor_inst_id));

      if (auto const_id = local_context()
                              .import_ir_constant_values()[cursor_ir_id.index]
                              .Get(cursor_inst_id);
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
    local_constant_values_for_import_insts().Set(inst_id, const_id);
    for (auto indirect_inst : indirect_insts) {
      local_context()
          .import_ir_constant_values()[indirect_inst.ir_id.index]
          .Set(indirect_inst.inst_id, const_id);
    }
  }

  auto PerformPendingWork() -> void;

  llvm::SmallVector<Work> work_stack_;
  // The size of work_stack_ at the start of resolving the current instruction.
  size_t initial_work_ = 0;
};
}  // namespace

static auto AddImportRef(ImportContext& context, SemIR::InstId inst_id,
                         SemIR::EntityNameId entity_name_id =
                             SemIR::EntityNameId::Invalid) -> SemIR::InstId {
  return AddImportRef(context.local_context(),
                      {.ir_id = context.import_ir_id(), .inst_id = inst_id},
                      entity_name_id);
}

static auto AddLoadedImportRef(ImportContext& context, SemIR::TypeId type_id,
                               SemIR::InstId inst_id,
                               SemIR::ConstantId const_id) -> SemIR::InstId {
  return AddLoadedImportRef(
      context.local_context(), type_id,
      {.ir_id = context.import_ir_id(), .inst_id = inst_id}, const_id);
}

static auto AddImportIRInst(ImportContext& context, SemIR::InstId inst_id)
    -> SemIR::ImportIRInstId {
  return context.local_import_ir_insts().Add(
      {.ir_id = context.import_ir_id(), .inst_id = inst_id});
}

// Returns the ConstantId for an InstId. Adds unresolved constants to
// the resolver's work stack.
static auto GetLocalConstantId(ImportRefResolver& resolver,
                               SemIR::InstId inst_id) -> SemIR::ConstantId {
  return resolver.GetLocalConstantValueOrPush(inst_id);
}

// Returns the ConstantId for an imported ConstantId. Adds unresolved
// constants to the resolver's work stack.
static auto GetLocalConstantId(ImportRefResolver& resolver,
                               SemIR::ConstantId const_id)
    -> SemIR::ConstantId {
  return GetLocalConstantId(
      resolver, GetInstWithConstantValue(resolver.import_ir(), const_id));
}

// Returns the local constant InstId for an imported InstId.
static auto GetLocalConstantInstId(ImportRefResolver& resolver,
                                   SemIR::InstId inst_id) -> SemIR::InstId {
  auto const_id = GetLocalConstantId(resolver, inst_id);
  return resolver.local_constant_values().GetInstIdIfValid(const_id);
}

// Returns the ConstantId for a TypeId. Adds unresolved constants to
// work_stack_.
static auto GetLocalConstantId(ImportRefResolver& resolver,
                               SemIR::TypeId type_id) -> SemIR::ConstantId {
  return GetLocalConstantId(resolver,
                            resolver.import_types().GetConstantId(type_id));
}

// Returns the ConstantId for an InstId that is required to have already been
// imported.
static auto GetLocalConstantIdChecked(ImportContext& context,
                                      SemIR::InstId inst_id)
    -> SemIR::ConstantId {
  auto result_id =
      context.local_constant_values_for_import_insts().Get(inst_id);
  CARBON_CHECK(result_id.is_valid());
  return result_id;
}

// Returns the ConstantId for a ConstantId that is required to have already been
// imported.
static auto GetLocalConstantIdChecked(ImportContext& context,
                                      SemIR::ConstantId const_id)
    -> SemIR::ConstantId {
  return GetLocalConstantIdChecked(
      context, GetInstWithConstantValue(context.import_ir(), const_id));
}

// Returns the ConstantId for a TypeId that is required to have already been
// imported.
static auto GetLocalConstantIdChecked(ImportContext& context,
                                      SemIR::TypeId type_id)
    -> SemIR::ConstantId {
  return GetLocalConstantIdChecked(
      context, context.import_types().GetConstantId(type_id));
}

// Translates a NameId from the import IR to a local NameId.
static auto GetLocalNameId(ImportContext& context, SemIR::NameId import_name_id)
    -> SemIR::NameId {
  if (auto ident_id = import_name_id.AsIdentifierId(); ident_id.is_valid()) {
    return SemIR::NameId::ForIdentifier(context.local_identifiers().Add(
        context.import_identifiers().Get(ident_id)));
  }
  return import_name_id;
}

// Gets the local constant values corresponding to an imported inst block.
static auto GetLocalInstBlockContents(ImportRefResolver& resolver,
                                      SemIR::InstBlockId import_block_id)
    -> llvm::SmallVector<SemIR::InstId> {
  llvm::SmallVector<SemIR::InstId> inst_ids;
  if (!import_block_id.is_valid() ||
      import_block_id == SemIR::InstBlockId::Empty) {
    return inst_ids;
  }

  // Import all the values in the block.
  auto import_block = resolver.import_inst_blocks().Get(import_block_id);
  inst_ids.reserve(import_block.size());
  for (auto import_inst_id : import_block) {
    auto const_id = GetLocalConstantId(resolver, import_inst_id);
    inst_ids.push_back(
        resolver.local_constant_values().GetInstIdIfValid(const_id));
  }

  return inst_ids;
}

// Gets a local canonical instruction block ID corresponding to an imported inst
// block whose contents were already imported, for example by
// GetLocalInstBlockContents.
static auto GetLocalCanonicalInstBlockId(ImportContext& context,
                                         SemIR::InstBlockId import_block_id,
                                         llvm::ArrayRef<SemIR::InstId> contents)
    -> SemIR::InstBlockId {
  if (!import_block_id.is_valid()) {
    return SemIR::InstBlockId::Invalid;
  }
  return context.local_inst_blocks().AddCanonical(contents);
}

// Gets an incomplete local version of an imported generic. Most fields are
// set in the third phase.
static auto MakeIncompleteGeneric(ImportContext& context, SemIR::InstId decl_id,
                                  SemIR::GenericId generic_id)
    -> SemIR::GenericId {
  if (!generic_id.is_valid()) {
    return SemIR::GenericId::Invalid;
  }

  return context.local_generics().Add(
      {.decl_id = decl_id,
       .bindings_id = SemIR::InstBlockId::Invalid,
       .self_specific_id = SemIR::SpecificId::Invalid});
}

namespace {
// Local information associated with an imported generic.
struct GenericData {
  llvm::SmallVector<SemIR::InstId> bindings;
};
}  // namespace

// Gets a local version of the data associated with a generic.
static auto GetLocalGenericData(ImportRefResolver& resolver,
                                SemIR::GenericId generic_id) -> GenericData {
  if (!generic_id.is_valid()) {
    return GenericData();
  }

  const auto& generic = resolver.import_generics().Get(generic_id);
  return {.bindings = GetLocalInstBlockContents(resolver, generic.bindings_id)};
}

// Adds the given local generic data to the given generic.
static auto SetGenericData(ImportContext& context,
                           SemIR::GenericId import_generic_id,
                           SemIR::GenericId new_generic_id,
                           const GenericData& generic_data) -> void {
  if (!import_generic_id.is_valid()) {
    return;
  }

  const auto& import_generic = context.import_generics().Get(import_generic_id);
  auto& new_generic = context.local_generics().Get(new_generic_id);
  new_generic.bindings_id = GetLocalCanonicalInstBlockId(
      context, import_generic.bindings_id, generic_data.bindings);

  // Track that we need to fill in the remaining information in
  // FinishPendingGeneric.
  context.AddPendingGeneric(
      {.import_id = import_generic_id, .local_id = new_generic_id});
}

// Gets a local constant value corresponding to an imported generic ID. May
// add work to the work stack and return `Invalid`.
static auto GetLocalConstantId(ImportRefResolver& resolver,
                               SemIR::GenericId generic_id)
    -> SemIR::ConstantId {
  if (!generic_id.is_valid()) {
    return SemIR::ConstantId::Invalid;
  }
  auto import_decl_inst_id = resolver.import_generics().Get(generic_id).decl_id;
  auto import_decl_inst = resolver.import_insts().Get(import_decl_inst_id);
  if (import_decl_inst.Is<SemIR::ImplDecl>()) {
    // For an impl declaration, the imported entity can be found via the
    // declaration.
    return GetLocalConstantId(resolver, import_decl_inst_id);
  }
  // For all other kinds of declaration, the imported entity can be found via
  // the type of the declaration.
  return GetLocalConstantId(resolver, import_decl_inst.type_id());
}

// Gets a local generic ID given the corresponding local constant ID returned
// by GetLocalConstantId for the imported generic. Does not add any new work.
static auto GetLocalGenericId(ImportContext& context,
                              SemIR::ConstantId local_const_id)
    -> SemIR::GenericId {
  if (!local_const_id.is_valid()) {
    return SemIR::GenericId::Invalid;
  }
  auto inst = context.local_insts().Get(
      context.local_constant_values().GetInstId(local_const_id));
  CARBON_KIND_SWITCH(inst) {
    case CARBON_KIND(SemIR::FunctionType fn_type): {
      return context.local_functions().Get(fn_type.function_id).generic_id;
    }
    case CARBON_KIND(SemIR::GenericClassType class_type): {
      return context.local_classes().Get(class_type.class_id).generic_id;
    }
    case CARBON_KIND(SemIR::GenericInterfaceType interface_type): {
      return context.local_interfaces()
          .Get(interface_type.interface_id)
          .generic_id;
    }
    case CARBON_KIND(SemIR::ImplDecl impl_decl): {
      return context.local_impls().Get(impl_decl.impl_id).generic_id;
    }
    default: {
      CARBON_FATAL("Unexpected inst for generic declaration: {0}", inst);
    }
  }
}

namespace {
// Local information associated with an imported specific.
struct SpecificData {
  SemIR::ConstantId generic_const_id;
  llvm::SmallVector<SemIR::InstId> args;
};
}  // namespace

// Gets local information about an imported specific.
static auto GetLocalSpecificData(ImportRefResolver& resolver,
                                 SemIR::SpecificId specific_id)
    -> SpecificData {
  if (!specific_id.is_valid()) {
    return {.generic_const_id = SemIR::ConstantId::Invalid, .args = {}};
  }

  const auto& specific = resolver.import_specifics().Get(specific_id);
  return {
      .generic_const_id = GetLocalConstantId(resolver, specific.generic_id),
      .args = GetLocalInstBlockContents(resolver, specific.args_id),
  };
}

// Gets a local specific whose data was already imported by
// GetLocalSpecificData. Does not add any new work.
static auto GetOrAddLocalSpecific(ImportContext& context,
                                  SemIR::SpecificId import_specific_id,
                                  const SpecificData& data)
    -> SemIR::SpecificId {
  if (!import_specific_id.is_valid()) {
    return SemIR::SpecificId::Invalid;
  }

  // Form a corresponding local specific ID.
  const auto& import_specific =
      context.import_specifics().Get(import_specific_id);
  auto generic_id = GetLocalGenericId(context, data.generic_const_id);
  auto args_id =
      GetLocalCanonicalInstBlockId(context, import_specific.args_id, data.args);

  // Get the specific.
  auto specific_id = context.local_specifics().GetOrAdd(generic_id, args_id);

  // Fill in the remaining information in FinishPendingSpecific, if necessary.
  auto& specific = context.local_specifics().Get(specific_id);
  if (!specific.decl_block_id.is_valid() ||
      (import_specific.definition_block_id.is_valid() &&
       !specific.definition_block_id.is_valid())) {
    context.AddPendingSpecific(
        {.import_id = import_specific_id, .local_id = specific_id});
  }
  return specific_id;
}

// Adds unresolved constants for each parameter's type to the resolver's work
// stack.
static auto LoadLocalPatternConstantIds(ImportRefResolver& resolver,
                                        SemIR::InstBlockId param_patterns_id)
    -> void {
  if (!param_patterns_id.is_valid() ||
      param_patterns_id == SemIR::InstBlockId::Empty) {
    return;
  }

  const auto& param_patterns =
      resolver.import_inst_blocks().Get(param_patterns_id);
  for (auto pattern_id : param_patterns) {
    auto pattern_inst = resolver.import_insts().Get(pattern_id);
    GetLocalConstantId(resolver, pattern_inst.type_id());
    if (auto addr = pattern_inst.TryAs<SemIR::AddrPattern>()) {
      pattern_id = addr->inner_id;
      pattern_inst = resolver.import_insts().Get(pattern_id);
      GetLocalConstantId(resolver, pattern_inst.type_id());
    }
    pattern_id = resolver.import_insts()
                     .GetAs<SemIR::ValueParamPattern>(pattern_id)
                     .subpattern_id;
    pattern_inst = resolver.import_insts().Get(pattern_id);
    // If the parameter is a symbolic binding, build the
    // SymbolicBindingPattern constant.
    if (pattern_inst.Is<SemIR::SymbolicBindingPattern>()) {
      GetLocalConstantId(resolver, pattern_id);
    }
  }
}

// Returns a version of param_patterns_id localized to the current IR.
//
// Must only be called after a call to
// LoadLocalPatternConstantIds(param_patterns_id) has completed without adding
// any new work to work_stack_.
//
// TODO: This is inconsistent with the rest of this class, which expects
// the relevant constants to be explicitly passed in. That makes it
// easier to statically detect when an input isn't loaded, but makes it
// harder to support importing more complex inst structures. We should
// take a holistic look at how to balance those concerns. For example,
// could the same function be used to load the constants and use them, with
// a parameter to select between the two?
static auto GetLocalParamPatternsId(ImportContext& context,
                                    SemIR::InstBlockId param_patterns_id)
    -> SemIR::InstBlockId {
  if (!param_patterns_id.is_valid() ||
      param_patterns_id == SemIR::InstBlockId::Empty) {
    return param_patterns_id;
  }
  const auto& param_patterns =
      context.import_inst_blocks().Get(param_patterns_id);
  llvm::SmallVector<SemIR::InstId> new_patterns;
  for (auto param_id : param_patterns) {
    // Figure out the pattern structure. This echoes
    // Function::GetParamPatternInfoFromPatternId.
    auto addr_pattern_id = param_id;
    auto addr_inst =
        context.import_insts().TryGetAs<SemIR::AddrPattern>(addr_pattern_id);
    auto param_pattern_id = addr_pattern_id;
    if (addr_inst) {
      param_pattern_id = addr_inst->inner_id;
    }

    auto param_pattern = context.import_insts().GetAs<SemIR::ValueParamPattern>(
        param_pattern_id);

    auto binding_id = param_pattern.subpattern_id;
    auto binding =
        context.import_insts().GetAs<SemIR::AnyBindingPattern>(binding_id);

    // Rebuild the pattern.
    auto entity_name =
        context.import_entity_names().Get(binding.entity_name_id);
    auto name_id = GetLocalNameId(context, entity_name.name_id);
    auto type_id = context.local_context().GetTypeIdForTypeConstant(
        GetLocalConstantIdChecked(context, binding.type_id));

    auto new_param_id = SemIR::InstId::Invalid;
    switch (binding.kind) {
      case SemIR::BindingPattern::Kind: {
        auto entity_name_id = context.local_entity_names().Add(
            {.name_id = name_id,
             .parent_scope_id = SemIR::NameScopeId::Invalid,
             .bind_index = SemIR::CompileTimeBindIndex::Invalid});
        new_param_id =
            context.local_context().AddInstInNoBlock<SemIR::BindingPattern>(
                AddImportIRInst(context, binding_id),
                {.type_id = type_id, .entity_name_id = entity_name_id});
        break;
      }
      case SemIR::SymbolicBindingPattern::Kind: {
        // We already imported a constant value for this symbolic binding.
        // We can reuse most of it, but update the value to point to our
        // specific parameter, and preserve the constant value.
        auto bind_const_id = GetLocalConstantIdChecked(context, binding_id);
        auto new_binding_inst =
            context.local_insts().GetAs<SemIR::SymbolicBindingPattern>(
                context.local_constant_values().GetInstId(bind_const_id));
        new_param_id = context.local_context().AddInstInNoBlock(
            AddImportIRInst(context, binding_id), new_binding_inst);
        context.local_constant_values().Set(new_param_id, bind_const_id);
        break;
      }
      default: {
        CARBON_FATAL("Unexpected kind: ", binding.kind);
      }
    }
    new_param_id = context.local_context().AddInstInNoBlock(
        context.local_context()
            .MakeImportedLocAndInst<SemIR::ValueParamPattern>(
                AddImportIRInst(context, param_pattern_id),
                {.type_id = type_id,
                 .subpattern_id = new_param_id,
                 .runtime_index = param_pattern.runtime_index}));
    if (addr_inst) {
      type_id = context.local_context().GetTypeIdForTypeConstant(
          GetLocalConstantIdChecked(context, addr_inst->type_id));
      new_param_id = context.local_context().AddInstInNoBlock(
          context.local_context().MakeImportedLocAndInst<SemIR::AddrPattern>(
              AddImportIRInst(context, addr_pattern_id),
              {.type_id = type_id, .inner_id = new_param_id}));
    }
    new_patterns.push_back(new_param_id);
  }
  return context.local_inst_blocks().Add(new_patterns);
}

// Returns a version of import_return_slot_pattern_id localized to the current
// IR.
static auto GetLocalReturnSlotPatternId(
    ImportContext& context, SemIR::InstId import_return_slot_pattern_id)
    -> SemIR::InstId {
  if (!import_return_slot_pattern_id.is_valid()) {
    return SemIR::InstId::Invalid;
  }

  auto param_pattern = context.import_insts().GetAs<SemIR::OutParamPattern>(
      import_return_slot_pattern_id);
  auto return_slot_pattern =
      context.import_insts().GetAs<SemIR::ReturnSlotPattern>(
          param_pattern.subpattern_id);
  auto type_id = context.local_context().GetTypeIdForTypeConstant(
      GetLocalConstantIdChecked(context, return_slot_pattern.type_id));

  auto new_return_slot_pattern_id = context.local_context().AddInstInNoBlock(
      context.local_context().MakeImportedLocAndInst<SemIR::ReturnSlotPattern>(
          AddImportIRInst(context, param_pattern.subpattern_id),
          {.type_id = type_id, .type_inst_id = SemIR::InstId::Invalid}));
  return context.local_context().AddInstInNoBlock(
      context.local_context().MakeImportedLocAndInst<SemIR::OutParamPattern>(
          AddImportIRInst(context, import_return_slot_pattern_id),
          {.type_id = type_id,
           .subpattern_id = new_return_slot_pattern_id,
           .runtime_index = param_pattern.runtime_index}));
}

// Translates a NameScopeId from the import IR to a local NameScopeId. Adds
// unresolved constants to the resolver's work stack.
static auto GetLocalNameScopeId(ImportRefResolver& resolver,
                                SemIR::NameScopeId name_scope_id)
    -> SemIR::NameScopeId {
  // Get the instruction that created the scope.
  auto [inst_id, inst] =
      resolver.import_name_scopes().GetInstIfValid(name_scope_id);
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
      const_id = resolver.local_constant_values_for_import_insts().Get(inst_id);
      break;

    default:
      const_id = GetLocalConstantId(resolver, inst_id);
  }
  if (!const_id.is_valid()) {
    return SemIR::NameScopeId::Invalid;
  }

  auto const_inst_id = resolver.local_constant_values().GetInstId(const_id);
  auto name_scope_inst = resolver.local_insts().Get(const_inst_id);
  CARBON_KIND_SWITCH(name_scope_inst) {
    case CARBON_KIND(SemIR::Namespace inst): {
      return inst.name_scope_id;
    }
    case CARBON_KIND(SemIR::ClassType inst): {
      return resolver.local_classes().Get(inst.class_id).scope_id;
    }
    case CARBON_KIND(SemIR::FacetType inst): {
      const SemIR::FacetTypeInfo& facet_type_info =
          resolver.local_facet_types().Get(inst.facet_type_id);
      // This is specifically the facet type produced by an interface
      // declaration, and so should consist of a single interface.
      // TODO: Will also have to handle named constraints here, once those are
      // implemented.
      auto interface = facet_type_info.TryAsSingleInterface();
      CARBON_CHECK(interface);
      return resolver.local_interfaces().Get(interface->interface_id).scope_id;
    }
    case CARBON_KIND(SemIR::ImplDecl inst): {
      return resolver.local_impls().Get(inst.impl_id).scope_id;
    }
    case SemIR::StructValue::Kind: {
      auto type_inst =
          resolver.local_types().GetAsInst(name_scope_inst.type_id());
      CARBON_KIND_SWITCH(type_inst) {
        case CARBON_KIND(SemIR::GenericClassType inst): {
          return resolver.local_classes().Get(inst.class_id).scope_id;
        }
        case CARBON_KIND(SemIR::GenericInterfaceType inst): {
          return resolver.local_interfaces().Get(inst.interface_id).scope_id;
        }
        default: {
          break;
        }
      }
      break;
    }
    default: {
      if (const_inst_id == SemIR::InstId::BuiltinErrorInst) {
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
static auto GetIncompleteLocalEntityBase(
    ImportContext& context, SemIR::InstId decl_id,
    const SemIR::EntityWithParamsBase& import_base)
    -> SemIR::EntityWithParamsBase {
  // Translate the extern_library_id if present.
  auto extern_library_id = SemIR::LibraryNameId::Invalid;
  if (import_base.extern_library_id.is_valid()) {
    if (import_base.extern_library_id.index >= 0) {
      auto val = context.import_string_literal_values().Get(
          import_base.extern_library_id.AsStringLiteralValueId());
      extern_library_id = SemIR::LibraryNameId::ForStringLiteralValueId(
          context.local_string_literal_values().Add(val));
    } else {
      extern_library_id = import_base.extern_library_id;
    }
  }

  return {
      .name_id = GetLocalNameId(context, import_base.name_id),
      .parent_scope_id = SemIR::NameScopeId::Invalid,
      .generic_id =
          MakeIncompleteGeneric(context, decl_id, import_base.generic_id),
      .first_param_node_id = Parse::NodeId::Invalid,
      .last_param_node_id = Parse::NodeId::Invalid,
      .pattern_block_id = SemIR::InstBlockId::Invalid,
      .implicit_param_patterns_id =
          import_base.implicit_param_patterns_id.is_valid()
              ? SemIR::InstBlockId::Empty
              : SemIR::InstBlockId::Invalid,
      .param_patterns_id = import_base.param_patterns_id.is_valid()
                               ? SemIR::InstBlockId::Empty
                               : SemIR::InstBlockId::Invalid,
      .call_params_id = SemIR::InstBlockId::Invalid,
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
static auto AddNameScopeImportRefs(ImportContext& context,
                                   const SemIR::NameScope& import_scope,
                                   SemIR::NameScope& new_scope) -> void {
  for (auto entry : import_scope.names) {
    auto ref_id = AddImportRef(context, entry.inst_id);
    new_scope.AddRequired({.name_id = GetLocalNameId(context, entry.name_id),
                           .inst_id = ref_id,
                           .access_kind = entry.access_kind});
  }
  for (auto scope_inst_id : import_scope.extended_scopes) {
    new_scope.extended_scopes.push_back(AddImportRef(context, scope_inst_id));
  }
}

// Given a block ID for a list of associated entities of a witness, returns a
// version localized to the current IR.
static auto AddAssociatedEntities(ImportContext& context,
                                  SemIR::NameScopeId local_name_scope_id,
                                  SemIR::InstBlockId associated_entities_id)
    -> SemIR::InstBlockId {
  if (associated_entities_id == SemIR::InstBlockId::Empty) {
    return SemIR::InstBlockId::Empty;
  }
  auto associated_entities =
      context.import_inst_blocks().Get(associated_entities_id);
  llvm::SmallVector<SemIR::InstId> new_associated_entities;
  new_associated_entities.reserve(associated_entities.size());
  for (auto inst_id : associated_entities) {
    // Determine the name of the associated entity, by switching on its type.
    SemIR::NameId import_name_id = SemIR::NameId::Invalid;
    if (auto associated_const =
            context.import_insts().TryGetAs<SemIR::AssociatedConstantDecl>(
                inst_id)) {
      import_name_id = associated_const->name_id;
    } else if (auto function_decl =
                   context.import_insts().TryGetAs<SemIR::FunctionDecl>(
                       inst_id)) {
      auto function =
          context.import_functions().Get(function_decl->function_id);
      import_name_id = function.name_id;
    } else {
      CARBON_CHECK("Unhandled associated entity type");
    }
    auto name_id = GetLocalNameId(context, import_name_id);
    auto entity_name_id = context.local_entity_names().Add(
        {.name_id = name_id,
         .parent_scope_id = local_name_scope_id,
         .bind_index = SemIR::CompileTimeBindIndex::Invalid});
    new_associated_entities.push_back(
        AddImportRef(context, inst_id, entity_name_id));
  }
  return context.local_inst_blocks().Add(new_associated_entities);
}

// Produces a resolve result that provides the given constant value. Retries
// instead if work has been added.
static auto RetryOrDone(ImportRefResolver& resolver, SemIR::ConstantId const_id)
    -> ResolveResult {
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  return ResolveResult::Done(const_id);
}

// Produces a resolve result for the given instruction that describes a constant
// value. This should only be used for instructions that describe constants, and
// not for instructions that represent declarations. For a declaration, we need
// an associated location, so AddInstInNoBlock should be used instead. Requires
// that there is no new work.
static auto ResolveAsUntyped(ImportContext& context, SemIR::Inst inst)
    -> ResolveResult {
  auto result =
      TryEvalInst(context.local_context(), SemIR::InstId::Invalid, inst);
  CARBON_CHECK(result.is_constant(), "{0} is not constant", inst);
  return ResolveResult::Done(result);
}

// Same as ResolveAsUntyped, but with an explicit type for convenience.
template <typename InstT>
static auto ResolveAs(ImportContext& context, InstT inst) -> ResolveResult {
  return ResolveAsUntyped(context, inst);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::AdaptDecl inst,
                                SemIR::InstId import_inst_id) -> ResolveResult {
  auto adapted_type_const_id = GetLocalConstantId(
      resolver,
      resolver.import_constant_values().Get(inst.adapted_type_inst_id));
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  auto adapted_type_inst_id =
      AddLoadedImportRef(resolver, SemIR::TypeId::TypeType,
                         inst.adapted_type_inst_id, adapted_type_const_id);

  // Create a corresponding instruction to represent the declaration.
  auto inst_id = resolver.local_context().AddInstInNoBlock(
      resolver.local_context().MakeImportedLocAndInst<SemIR::AdaptDecl>(
          AddImportIRInst(resolver, import_inst_id),
          {.adapted_type_inst_id = adapted_type_inst_id}));
  return ResolveResult::Done(resolver.local_constant_values().Get(inst_id));
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::AssociatedEntity inst) -> ResolveResult {
  auto type_const_id = GetLocalConstantId(resolver, inst.type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  // Add a lazy reference to the target declaration.
  auto decl_id = AddImportRef(resolver, inst.decl_id);

  return ResolveAs<SemIR::AssociatedEntity>(
      resolver, {.type_id = resolver.local_context().GetTypeIdForTypeConstant(
                     type_const_id),
                 .index = inst.index,
                 .decl_id = decl_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::AssociatedEntityType inst)
    -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

  auto entity_type_const_id = GetLocalConstantId(resolver, inst.entity_type_id);
  auto interface_inst_id = GetLocalConstantId(resolver, inst.interface_type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::AssociatedEntityType>(
      resolver,
      {.type_id = SemIR::TypeId::TypeType,
       .interface_type_id =
           resolver.local_context().GetTypeIdForTypeConstant(interface_inst_id),
       .entity_type_id = resolver.local_context().GetTypeIdForTypeConstant(
           entity_type_const_id)});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::BaseDecl inst,
                                SemIR::InstId import_inst_id) -> ResolveResult {
  auto type_const_id = GetLocalConstantId(resolver, inst.type_id);
  auto base_type_const_id = GetLocalConstantId(
      resolver, resolver.import_constant_values().Get(inst.base_type_inst_id));
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  auto base_type_inst_id =
      AddLoadedImportRef(resolver, SemIR::TypeId::TypeType,
                         inst.base_type_inst_id, base_type_const_id);

  // Create a corresponding instruction to represent the declaration.
  auto inst_id = resolver.local_context().AddInstInNoBlock(
      resolver.local_context().MakeImportedLocAndInst<SemIR::BaseDecl>(
          AddImportIRInst(resolver, import_inst_id),
          {.type_id =
               resolver.local_context().GetTypeIdForTypeConstant(type_const_id),
           .base_type_inst_id = base_type_inst_id,
           .index = inst.index}));
  return ResolveResult::Done(resolver.local_constant_values().Get(inst_id));
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::BindAlias inst) -> ResolveResult {
  auto value_id = GetLocalConstantId(resolver, inst.value_id);
  return RetryOrDone(resolver, value_id);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::BindSymbolicName inst) -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  const auto& import_entity_name =
      resolver.import_entity_names().Get(inst.entity_name_id);
  auto name_id = GetLocalNameId(resolver, import_entity_name.name_id);
  auto entity_name_id = resolver.local_entity_names().Add(
      {.name_id = name_id,
       .parent_scope_id = SemIR::NameScopeId::Invalid,
       .bind_index = import_entity_name.bind_index});
  return ResolveAs<SemIR::BindSymbolicName>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .entity_name_id = entity_name_id,
       .value_id = SemIR::InstId::Invalid});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::SymbolicBindingPattern inst)
    -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  const auto& import_entity_name =
      resolver.import_entity_names().Get(inst.entity_name_id);
  auto name_id = GetLocalNameId(resolver, import_entity_name.name_id);
  auto entity_name_id = resolver.local_entity_names().Add(
      {.name_id = name_id,
       .parent_scope_id = SemIR::NameScopeId::Invalid,
       .bind_index = import_entity_name.bind_index});
  return ResolveAs<SemIR::SymbolicBindingPattern>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .entity_name_id = entity_name_id});
}

// Makes an incomplete class. This is necessary even with classes with a
// complete declaration, because things such as `Self` may refer back to the
// type.
static auto MakeIncompleteClass(ImportContext& context,
                                const SemIR::Class& import_class,
                                SemIR::SpecificId enclosing_specific_id)
    -> std::pair<SemIR::ClassId, SemIR::ConstantId> {
  SemIR::ClassDecl class_decl = {.type_id = SemIR::TypeId::TypeType,
                                 .class_id = SemIR::ClassId::Invalid,
                                 .decl_block_id = SemIR::InstBlockId::Empty};
  auto class_decl_id = context.local_context().AddPlaceholderInstInNoBlock(
      context.local_context().MakeImportedLocAndInst(
          AddImportIRInst(context, import_class.latest_decl_id()), class_decl));
  // Regardless of whether ClassDecl is a complete type, we first need an
  // incomplete type so that any references have something to point at.
  class_decl.class_id = context.local_classes().Add(
      {GetIncompleteLocalEntityBase(context, class_decl_id, import_class),
       {.self_type_id = SemIR::TypeId::Invalid,
        .inheritance_kind = import_class.inheritance_kind,
        .is_dynamic = import_class.is_dynamic}});

  if (import_class.has_parameters()) {
    class_decl.type_id = context.local_context().GetGenericClassType(
        class_decl.class_id, enclosing_specific_id);
  }

  // Write the class ID into the ClassDecl.
  context.local_context().ReplaceInstBeforeConstantUse(class_decl_id,
                                                       class_decl);
  auto self_const_id = context.local_constant_values().Get(class_decl_id);
  return {class_decl.class_id, self_const_id};
}

// Fills out the class definition for an incomplete class.
static auto AddClassDefinition(ImportContext& context,
                               const SemIR::Class& import_class,
                               SemIR::Class& new_class,
                               SemIR::InstId complete_type_witness_id,
                               SemIR::InstId base_id, SemIR::InstId adapt_id)
    -> void {
  new_class.definition_id = new_class.first_owning_decl_id;

  new_class.complete_type_witness_id = complete_type_witness_id;

  new_class.scope_id = context.local_name_scopes().Add(
      new_class.first_owning_decl_id, SemIR::NameId::Invalid,
      new_class.parent_scope_id);
  auto& new_scope = context.local_name_scopes().Get(new_class.scope_id);
  const auto& import_scope =
      context.import_name_scopes().Get(import_class.scope_id);

  // Push a block so that we can add scoped instructions to it.
  context.local_context().inst_block_stack().Push();
  AddNameScopeImportRefs(context, import_scope, new_scope);
  new_class.body_block_id = context.local_context().inst_block_stack().Pop();

  if (import_class.base_id.is_valid()) {
    new_class.base_id = base_id;
  }
  if (import_class.adapt_id.is_valid()) {
    new_class.adapt_id = adapt_id;
  }
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::ClassDecl inst,
                                SemIR::ConstantId class_const_id)
    -> ResolveResult {
  // TODO: The handling of interfaces repeats a lot with the handling of
  // classes, and will likely also be repeated for named constraints and
  // choice types. Factor out some of this functionality.
  const auto& import_class = resolver.import_classes().Get(inst.class_id);

  SemIR::ClassId class_id = SemIR::ClassId::Invalid;
  if (!class_const_id.is_valid()) {
    auto import_specific_id = SemIR::SpecificId::Invalid;
    if (auto import_generic_class_type =
            resolver.import_types().TryGetAs<SemIR::GenericClassType>(
                inst.type_id)) {
      import_specific_id = import_generic_class_type->enclosing_specific_id;
    }
    auto specific_data = GetLocalSpecificData(resolver, import_specific_id);
    if (resolver.HasNewWork()) {
      // This is the end of the first phase. Don't make a new class yet if
      // we already have new work.
      return ResolveResult::Retry();
    }

    // On the second phase, create a forward declaration of the class for any
    // recursive references.
    auto enclosing_specific_id =
        GetOrAddLocalSpecific(resolver, import_specific_id, specific_data);
    std::tie(class_id, class_const_id) =
        MakeIncompleteClass(resolver, import_class, enclosing_specific_id);
  } else {
    // On the third phase, compute the class ID from the constant
    // value of the declaration.
    auto class_const_inst = resolver.local_insts().Get(
        resolver.local_constant_values().GetInstId(class_const_id));
    if (auto class_type = class_const_inst.TryAs<SemIR::ClassType>()) {
      class_id = class_type->class_id;
    } else {
      auto generic_class_type =
          resolver.local_types().GetAs<SemIR::GenericClassType>(
              class_const_inst.type_id());
      class_id = generic_class_type.class_id;
    }
  }

  // Load constants for the definition.
  auto parent_scope_id =
      GetLocalNameScopeId(resolver, import_class.parent_scope_id);
  LoadLocalPatternConstantIds(resolver,
                              import_class.implicit_param_patterns_id);
  LoadLocalPatternConstantIds(resolver, import_class.param_patterns_id);
  auto generic_data = GetLocalGenericData(resolver, import_class.generic_id);
  auto self_const_id = GetLocalConstantId(resolver, import_class.self_type_id);
  auto complete_type_witness_const_id =
      import_class.complete_type_witness_id.is_valid()
          ? GetLocalConstantId(resolver, import_class.complete_type_witness_id)
          : SemIR::ConstantId::Invalid;
  auto base_id = import_class.base_id.is_valid()
                     ? GetLocalConstantInstId(resolver, import_class.base_id)
                     : SemIR::InstId::Invalid;
  auto adapt_id = import_class.adapt_id.is_valid()
                      ? GetLocalConstantInstId(resolver, import_class.adapt_id)
                      : SemIR::InstId::Invalid;

  if (resolver.HasNewWork()) {
    return ResolveResult::Retry(class_const_id);
  }

  auto& new_class = resolver.local_classes().Get(class_id);
  new_class.parent_scope_id = parent_scope_id;
  new_class.implicit_param_patterns_id = GetLocalParamPatternsId(
      resolver, import_class.implicit_param_patterns_id);
  new_class.param_patterns_id =
      GetLocalParamPatternsId(resolver, import_class.param_patterns_id);
  SetGenericData(resolver, import_class.generic_id, new_class.generic_id,
                 generic_data);
  new_class.self_type_id =
      resolver.local_context().GetTypeIdForTypeConstant(self_const_id);

  if (import_class.is_defined()) {
    auto complete_type_witness_id = AddLoadedImportRef(
        resolver,
        resolver.local_context().GetBuiltinType(
            SemIR::BuiltinInstKind::WitnessType),
        import_class.complete_type_witness_id, complete_type_witness_const_id);
    AddClassDefinition(resolver, import_class, new_class,
                       complete_type_witness_id, base_id, adapt_id);
  }

  return ResolveResult::Done(class_const_id);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::ClassType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto class_const_id = GetLocalConstantId(
      resolver,
      resolver.import_classes().Get(inst.class_id).first_owning_decl_id);
  auto specific_data = GetLocalSpecificData(resolver, inst.specific_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  // Find the corresponding class type. For a non-generic class, this is the
  // type of the class declaration. For a generic class, build a class type
  // referencing this specialization of the generic class.
  auto class_const_inst = resolver.local_insts().Get(
      resolver.local_constant_values().GetInstId(class_const_id));
  if (class_const_inst.Is<SemIR::ClassType>()) {
    return ResolveResult::Done(class_const_id);
  } else {
    auto generic_class_type =
        resolver.local_types().GetAs<SemIR::GenericClassType>(
            class_const_inst.type_id());
    auto specific_id =
        GetOrAddLocalSpecific(resolver, inst.specific_id, specific_data);
    return ResolveAs<SemIR::ClassType>(resolver,
                                       {.type_id = SemIR::TypeId::TypeType,
                                        .class_id = generic_class_type.class_id,
                                        .specific_id = specific_id});
  }
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::CompleteTypeWitness inst)
    -> ResolveResult {
  CARBON_CHECK(resolver.import_types().GetInstId(inst.type_id) ==
               SemIR::InstId::BuiltinWitnessType);
  auto object_repr_const_id = GetLocalConstantId(resolver, inst.object_repr_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  auto object_repr_id =
      resolver.local_context().GetTypeIdForTypeConstant(object_repr_const_id);
  return ResolveAs<SemIR::CompleteTypeWitness>(
      resolver, {.type_id = resolver.local_context().GetBuiltinType(
                     SemIR::BuiltinInstKind::WitnessType),
                 .object_repr_id = object_repr_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::ConstType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto inner_const_id = GetLocalConstantId(resolver, inst.inner_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  auto inner_type_id =
      resolver.local_context().GetTypeIdForTypeConstant(inner_const_id);
  return ResolveAs<SemIR::ConstType>(
      resolver,
      {.type_id = SemIR::TypeId::TypeType, .inner_id = inner_type_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::ExportDecl inst) -> ResolveResult {
  auto value_id = GetLocalConstantId(resolver, inst.value_id);
  return RetryOrDone(resolver, value_id);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FieldDecl inst,
                                SemIR::InstId import_inst_id) -> ResolveResult {
  auto const_id = GetLocalConstantId(resolver, inst.type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  auto inst_id = resolver.local_context().AddInstInNoBlock(
      resolver.local_context().MakeImportedLocAndInst<SemIR::FieldDecl>(
          AddImportIRInst(resolver, import_inst_id),
          {.type_id =
               resolver.local_context().GetTypeIdForTypeConstant(const_id),
           .name_id = GetLocalNameId(resolver, inst.name_id),
           .index = inst.index}));
  return {.const_id = resolver.local_constant_values().Get(inst_id)};
}

// Make a declaration of a function. This is done as a separate step from
// importing the function declaration in order to resolve cycles.
static auto MakeFunctionDecl(ImportContext& context,
                             const SemIR::Function& import_function,
                             SemIR::SpecificId specific_id)
    -> std::pair<SemIR::FunctionId, SemIR::ConstantId> {
  SemIR::FunctionDecl function_decl = {
      .type_id = SemIR::TypeId::Invalid,
      .function_id = SemIR::FunctionId::Invalid,
      .decl_block_id = SemIR::InstBlockId::Empty};
  auto function_decl_id = context.local_context().AddPlaceholderInstInNoBlock(
      context.local_context().MakeImportedLocAndInst(
          AddImportIRInst(context, import_function.first_decl_id()),
          function_decl));

  // Start with an incomplete function.
  function_decl.function_id = context.local_functions().Add(
      {GetIncompleteLocalEntityBase(context, function_decl_id, import_function),
       {.return_slot_pattern_id = SemIR::InstId::Invalid,
        .builtin_function_kind = import_function.builtin_function_kind}});

  function_decl.type_id = context.local_context().GetFunctionType(
      function_decl.function_id, specific_id);

  // Write the function ID and type into the FunctionDecl.
  context.local_context().ReplaceInstBeforeConstantUse(function_decl_id,
                                                       function_decl);
  return {function_decl.function_id,
          context.local_constant_values().Get(function_decl_id)};
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FunctionDecl inst,
                                SemIR::ConstantId function_const_id)
    -> ResolveResult {
  const auto& import_function =
      resolver.import_functions().Get(inst.function_id);

  SemIR::FunctionId function_id = SemIR::FunctionId::Invalid;
  if (!function_const_id.is_valid()) {
    auto import_specific_id = resolver.import_types()
                                  .GetAs<SemIR::FunctionType>(inst.type_id)
                                  .specific_id;
    auto specific_data = GetLocalSpecificData(resolver, import_specific_id);
    if (resolver.HasNewWork()) {
      // This is the end of the first phase. Don't make a new function yet if
      // we already have new work.
      return ResolveResult::Retry();
    }

    // On the second phase, create a forward declaration of the interface.
    auto specific_id =
        GetOrAddLocalSpecific(resolver, import_specific_id, specific_data);
    std::tie(function_id, function_const_id) =
        MakeFunctionDecl(resolver, import_function, specific_id);
  } else {
    // On the third phase, compute the function ID from the constant value of
    // the declaration.
    auto function_const_inst = resolver.local_insts().Get(
        resolver.local_constant_values().GetInstId(function_const_id));
    auto function_type = resolver.local_types().GetAs<SemIR::FunctionType>(
        function_const_inst.type_id());
    function_id = function_type.function_id;
  }

  auto return_type_const_id = SemIR::ConstantId::Invalid;
  if (import_function.return_slot_pattern_id.is_valid()) {
    return_type_const_id = GetLocalConstantId(
        resolver, resolver.import_insts()
                      .Get(import_function.return_slot_pattern_id)
                      .type_id());
  }
  auto parent_scope_id =
      GetLocalNameScopeId(resolver, import_function.parent_scope_id);
  LoadLocalPatternConstantIds(resolver,
                              import_function.implicit_param_patterns_id);
  LoadLocalPatternConstantIds(resolver, import_function.param_patterns_id);
  auto generic_data = GetLocalGenericData(resolver, import_function.generic_id);

  if (resolver.HasNewWork()) {
    return ResolveResult::Retry(function_const_id);
  }

  // Add the function declaration.
  auto& new_function = resolver.local_functions().Get(function_id);
  new_function.parent_scope_id = parent_scope_id;
  new_function.implicit_param_patterns_id = GetLocalParamPatternsId(
      resolver, import_function.implicit_param_patterns_id);
  new_function.param_patterns_id =
      GetLocalParamPatternsId(resolver, import_function.param_patterns_id);
  new_function.return_slot_pattern_id = GetLocalReturnSlotPatternId(
      resolver, import_function.return_slot_pattern_id);
  SetGenericData(resolver, import_function.generic_id, new_function.generic_id,
                 generic_data);

  if (import_function.definition_id.is_valid()) {
    new_function.definition_id = new_function.first_owning_decl_id;
  }

  return ResolveResult::Done(function_const_id);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FunctionType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto fn_val_id = GetLocalConstantInstId(
      resolver,
      resolver.import_functions().Get(inst.function_id).first_decl_id());
  auto specific_data = GetLocalSpecificData(resolver, inst.specific_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  auto fn_type_id = resolver.local_insts().Get(fn_val_id).type_id();
  return ResolveAs<SemIR::FunctionType>(
      resolver, {.type_id = SemIR::TypeId::TypeType,
                 .function_id = resolver.local_types()
                                    .GetAs<SemIR::FunctionType>(fn_type_id)
                                    .function_id,
                 .specific_id = GetOrAddLocalSpecific(
                     resolver, inst.specific_id, specific_data)});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::GenericClassType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto class_val_id = GetLocalConstantInstId(
      resolver,
      resolver.import_classes().Get(inst.class_id).first_owning_decl_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  auto class_val = resolver.local_insts().Get(class_val_id);
  CARBON_CHECK(
      resolver.local_types().Is<SemIR::GenericClassType>(class_val.type_id()));
  return ResolveResult::Done(
      resolver.local_types().GetConstantId(class_val.type_id()));
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::GenericInterfaceType inst)
    -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto interface_val_id = GetLocalConstantInstId(
      resolver,
      resolver.import_interfaces().Get(inst.interface_id).first_owning_decl_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }
  auto interface_val = resolver.local_insts().Get(interface_val_id);
  CARBON_CHECK(resolver.local_types().Is<SemIR::GenericInterfaceType>(
      interface_val.type_id()));
  return ResolveResult::Done(
      resolver.local_types().GetConstantId(interface_val.type_id()));
}

// Make a declaration of an impl. This is done as a separate step from
// importing the impl definition in order to resolve cycles.
static auto MakeImplDeclaration(ImportContext& context,
                                const SemIR::Impl& import_impl)
    -> std::pair<SemIR::ImplId, SemIR::ConstantId> {
  SemIR::ImplDecl impl_decl = {.impl_id = SemIR::ImplId::Invalid,
                               .decl_block_id = SemIR::InstBlockId::Empty};
  auto impl_decl_id = context.local_context().AddPlaceholderInstInNoBlock(
      context.local_context().MakeImportedLocAndInst(
          AddImportIRInst(context, import_impl.latest_decl_id()), impl_decl));
  impl_decl.impl_id = context.local_impls().Add(
      {GetIncompleteLocalEntityBase(context, impl_decl_id, import_impl),
       {.self_id = SemIR::InstId::Invalid,
        .constraint_id = SemIR::InstId::Invalid,
        .witness_id = SemIR::InstId::Invalid}});

  // Write the impl ID into the ImplDecl.
  context.local_context().ReplaceInstBeforeConstantUse(impl_decl_id, impl_decl);
  return {impl_decl.impl_id, context.local_constant_values().Get(impl_decl_id)};
}

// Imports the definition of an impl.
static auto AddImplDefinition(ImportContext& context,
                              const SemIR::Impl& import_impl,
                              SemIR::Impl& new_impl, SemIR::InstId witness_id)
    -> void {
  new_impl.definition_id = new_impl.first_owning_decl_id;

  new_impl.witness_id = witness_id;

  if (import_impl.scope_id.is_valid()) {
    new_impl.scope_id = context.local_name_scopes().Add(
        new_impl.first_owning_decl_id, SemIR::NameId::Invalid,
        new_impl.parent_scope_id);
    // Import the contents of the definition scope, if we might need it. Name
    // lookup is never performed into this scope by a user of the impl, so
    // this is only necessary in the same library that defined the impl, in
    // order to support defining members of the impl out of line in the impl
    // file when the impl is defined in the API file.
    // TODO: Check to see if this impl is owned by the API file, rather than
    // merely being imported into it.
    if (context.import_ir_id() == SemIR::ImportIRId::ApiForImpl) {
      auto& new_scope = context.local_name_scopes().Get(new_impl.scope_id);
      const auto& import_scope =
          context.import_name_scopes().Get(import_impl.scope_id);

      // Push a block so that we can add scoped instructions to it.
      context.local_context().inst_block_stack().Push();
      AddNameScopeImportRefs(context, import_scope, new_scope);
      new_impl.body_block_id = context.local_context().inst_block_stack().Pop();
    }
  }
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::ImplDecl inst,
                                SemIR::ConstantId impl_const_id)
    -> ResolveResult {
  // TODO: This duplicates a lot of the handling of interfaces, classes, and
  // functions. Factor out the commonality.
  const auto& import_impl = resolver.import_impls().Get(inst.impl_id);

  SemIR::ImplId impl_id = SemIR::ImplId::Invalid;
  if (!impl_const_id.is_valid()) {
    if (resolver.HasNewWork()) {
      // This is the end of the first phase. Don't make a new impl yet if we
      // already have new work.
      return ResolveResult::Retry();
    }

    // On the second phase, create a forward declaration of the impl for any
    // recursive references.
    std::tie(impl_id, impl_const_id) =
        MakeImplDeclaration(resolver, import_impl);
  } else {
    // On the third phase, compute the impl ID from the "constant value" of
    // the declaration, which is a reference to the created ImplDecl.
    auto impl_const_inst = resolver.local_insts().GetAs<SemIR::ImplDecl>(
        resolver.local_constant_values().GetInstId(impl_const_id));
    impl_id = impl_const_inst.impl_id;
  }

  // Load constants for the definition.
  auto parent_scope_id =
      GetLocalNameScopeId(resolver, import_impl.parent_scope_id);
  LoadLocalPatternConstantIds(resolver, import_impl.implicit_param_patterns_id);
  auto generic_data = GetLocalGenericData(resolver, import_impl.generic_id);
  auto self_const_id = GetLocalConstantId(
      resolver, resolver.import_constant_values().Get(import_impl.self_id));
  auto constraint_const_id = GetLocalConstantId(
      resolver,
      resolver.import_constant_values().Get(import_impl.constraint_id));

  if (resolver.HasNewWork()) {
    return ResolveResult::Retry(impl_const_id);
  }

  auto& new_impl = resolver.local_impls().Get(impl_id);
  new_impl.parent_scope_id = parent_scope_id;
  new_impl.implicit_param_patterns_id =
      GetLocalParamPatternsId(resolver, import_impl.implicit_param_patterns_id);
  SetGenericData(resolver, import_impl.generic_id, new_impl.generic_id,
                 generic_data);

  // Create instructions for self and constraint to hold the symbolic constant
  // value for a generic impl.
  new_impl.self_id = AddLoadedImportRef(resolver, SemIR::TypeId::TypeType,
                                        import_impl.self_id, self_const_id);
  new_impl.constraint_id =
      AddLoadedImportRef(resolver, SemIR::TypeId::TypeType,
                         import_impl.constraint_id, constraint_const_id);

  if (import_impl.is_defined()) {
    auto witness_id = AddImportRef(resolver, import_impl.witness_id);
    AddImplDefinition(resolver, import_impl, new_impl, witness_id);
  }

  // If the `impl` is declared in the API file corresponding to the current
  // file, add this to impl lookup so that it can be found by redeclarations
  // in the current file.
  if (resolver.import_ir_id() == SemIR::ImportIRId::ApiForImpl) {
    resolver.local_impls().GetOrAddLookupBucket(new_impl).push_back(impl_id);
  }

  return ResolveResult::Done(impl_const_id);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::ImportRefLoaded /*inst*/,
                                SemIR::InstId inst_id) -> ResolveResult {
  // Return the constant for the instruction of the imported constant.
  auto constant_id = resolver.import_constant_values().Get(inst_id);
  if (!constant_id.is_valid()) {
    return ResolveResult::Done(SemIR::ConstantId::Error);
  }
  if (!constant_id.is_constant()) {
    resolver.local_context().TODO(
        inst_id, "Non-constant ImportRefLoaded (comes up with var)");
    return ResolveResult::Done(SemIR::ConstantId::Error);
  }

  auto new_constant_id = GetLocalConstantId(
      resolver, resolver.import_constant_values().GetInstId(constant_id));
  return RetryOrDone(resolver, new_constant_id);
}

// Make a declaration of an interface. This is done as a separate step from
// importing the interface definition in order to resolve cycles.
static auto MakeInterfaceDecl(ImportContext& context,
                              const SemIR::Interface& import_interface,
                              SemIR::SpecificId enclosing_specific_id)
    -> std::pair<SemIR::InterfaceId, SemIR::ConstantId> {
  SemIR::InterfaceDecl interface_decl = {
      .type_id = SemIR::TypeId::TypeType,
      .interface_id = SemIR::InterfaceId::Invalid,
      .decl_block_id = SemIR::InstBlockId::Empty};
  auto interface_decl_id = context.local_context().AddPlaceholderInstInNoBlock(
      context.local_context().MakeImportedLocAndInst(
          AddImportIRInst(context, import_interface.first_owning_decl_id),
          interface_decl));

  // Start with an incomplete interface.
  interface_decl.interface_id = context.local_interfaces().Add(
      {GetIncompleteLocalEntityBase(context, interface_decl_id,
                                    import_interface),
       {}});

  if (import_interface.has_parameters()) {
    interface_decl.type_id = context.local_context().GetGenericInterfaceType(
        interface_decl.interface_id, enclosing_specific_id);
  }

  // Write the interface ID into the InterfaceDecl.
  context.local_context().ReplaceInstBeforeConstantUse(interface_decl_id,
                                                       interface_decl);
  return {interface_decl.interface_id,
          context.local_constant_values().Get(interface_decl_id)};
}

// Imports the definition for an interface that has been imported as a forward
// declaration.
static auto AddInterfaceDefinition(ImportContext& context,
                                   const SemIR::Interface& import_interface,
                                   SemIR::Interface& new_interface,
                                   SemIR::InstId self_param_id) -> void {
  new_interface.scope_id = context.local_name_scopes().Add(
      new_interface.first_owning_decl_id, SemIR::NameId::Invalid,
      new_interface.parent_scope_id);
  auto& new_scope = context.local_name_scopes().Get(new_interface.scope_id);
  const auto& import_scope =
      context.import_name_scopes().Get(import_interface.scope_id);

  // Push a block so that we can add scoped instructions to it.
  context.local_context().inst_block_stack().Push();
  AddNameScopeImportRefs(context, import_scope, new_scope);
  new_interface.associated_entities_id = AddAssociatedEntities(
      context, new_interface.scope_id, import_interface.associated_entities_id);
  new_interface.body_block_id =
      context.local_context().inst_block_stack().Pop();
  new_interface.self_param_id = self_param_id;

  CARBON_CHECK(import_scope.extended_scopes.empty(),
               "Interfaces don't currently have extended scopes to support.");
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::InterfaceDecl inst,
                                SemIR::ConstantId interface_const_id)
    -> ResolveResult {
  const auto& import_interface =
      resolver.import_interfaces().Get(inst.interface_id);

  SemIR::InterfaceId interface_id = SemIR::InterfaceId::Invalid;
  if (!interface_const_id.is_valid()) {
    auto import_specific_id = SemIR::SpecificId::Invalid;
    if (auto import_generic_interface_type =
            resolver.import_types().TryGetAs<SemIR::GenericInterfaceType>(
                inst.type_id)) {
      import_specific_id = import_generic_interface_type->enclosing_specific_id;
    }
    auto specific_data = GetLocalSpecificData(resolver, import_specific_id);
    if (resolver.HasNewWork()) {
      // This is the end of the first phase. Don't make a new interface yet if
      // we already have new work.
      return ResolveResult::Retry();
    }

    // On the second phase, create a forward declaration of the interface.
    auto enclosing_specific_id =
        GetOrAddLocalSpecific(resolver, import_specific_id, specific_data);
    std::tie(interface_id, interface_const_id) =
        MakeInterfaceDecl(resolver, import_interface, enclosing_specific_id);
  } else {
    // On the third phase, compute the interface ID from the constant value of
    // the declaration.
    auto interface_const_inst = resolver.local_insts().Get(
        resolver.local_constant_values().GetInstId(interface_const_id));
    if (auto facet_type = interface_const_inst.TryAs<SemIR::FacetType>()) {
      const SemIR::FacetTypeInfo& facet_type_info =
          resolver.local_facet_types().Get(facet_type->facet_type_id);
      auto interface_type = facet_type_info.TryAsSingleInterface();
      CARBON_CHECK(interface_type);
      interface_id = interface_type->interface_id;
    } else {
      auto generic_interface_type =
          resolver.local_types().GetAs<SemIR::GenericInterfaceType>(
              interface_const_inst.type_id());
      interface_id = generic_interface_type.interface_id;
    }
  }

  auto parent_scope_id =
      GetLocalNameScopeId(resolver, import_interface.parent_scope_id);
  LoadLocalPatternConstantIds(resolver,
                              import_interface.implicit_param_patterns_id);
  LoadLocalPatternConstantIds(resolver, import_interface.param_patterns_id);
  auto generic_data =
      GetLocalGenericData(resolver, import_interface.generic_id);

  std::optional<SemIR::InstId> self_param_id;
  if (import_interface.is_defined()) {
    self_param_id =
        GetLocalConstantInstId(resolver, import_interface.self_param_id);
  }

  if (resolver.HasNewWork()) {
    return ResolveResult::Retry(interface_const_id);
  }

  auto& new_interface = resolver.local_interfaces().Get(interface_id);
  new_interface.parent_scope_id = parent_scope_id;
  new_interface.implicit_param_patterns_id = GetLocalParamPatternsId(
      resolver, import_interface.implicit_param_patterns_id);
  new_interface.param_patterns_id =
      GetLocalParamPatternsId(resolver, import_interface.param_patterns_id);
  SetGenericData(resolver, import_interface.generic_id,
                 new_interface.generic_id, generic_data);

  if (import_interface.is_defined()) {
    CARBON_CHECK(self_param_id);
    AddInterfaceDefinition(resolver, import_interface, new_interface,
                           *self_param_id);
  }
  return ResolveResult::Done(interface_const_id);
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FacetAccessType inst) -> ResolveResult {
  auto facet_value_inst_id =
      GetLocalConstantInstId(resolver, inst.facet_value_inst_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::FacetAccessType>(
      resolver, {.type_id = SemIR::TypeId::TypeType,
                 .facet_value_inst_id = facet_value_inst_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FacetAccessWitness inst)
    -> ResolveResult {
  auto facet_value_inst_id =
      GetLocalConstantInstId(resolver, inst.facet_value_inst_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::FacetAccessWitness>(
      resolver, {.type_id = resolver.local_context().GetBuiltinType(
                     SemIR::BuiltinInstKind::WitnessType),
                 .facet_value_inst_id = facet_value_inst_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FacetType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

  const SemIR::FacetTypeInfo& facet_type_info =
      resolver.import_facet_types().Get(inst.facet_type_id);
  for (auto interface : facet_type_info.impls_constraints) {
    GetLocalConstantId(resolver, resolver.import_interfaces()
                                     .Get(interface.interface_id)
                                     .first_owning_decl_id);
    GetLocalSpecificData(resolver, interface.specific_id);
  }
  for (auto rewrite : facet_type_info.rewrite_constraints) {
    GetLocalConstantId(resolver, rewrite.lhs_const_id);
    GetLocalConstantId(resolver, rewrite.rhs_const_id);
  }
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  llvm::SmallVector<SemIR::FacetTypeInfo::ImplsConstraint> impls_constraints;
  for (auto interface : facet_type_info.impls_constraints) {
    auto interface_const_id =
        GetLocalConstantId(resolver, resolver.import_interfaces()
                                         .Get(interface.interface_id)
                                         .first_owning_decl_id);
    auto specific_data = GetLocalSpecificData(resolver, interface.specific_id);

    // Find the corresponding interface type. For a non-generic interface,
    // this is the type of the interface declaration. For a generic interface,
    // build a interface type referencing this specialization of the generic
    // interface.
    auto interface_const_inst = resolver.local_insts().Get(
        resolver.local_constant_values().GetInstId(interface_const_id));
    if (auto facet_type = interface_const_inst.TryAs<SemIR::FacetType>()) {
      const SemIR::FacetTypeInfo& new_facet_type_info =
          resolver.local_facet_types().Get(facet_type->facet_type_id);
      impls_constraints.append(new_facet_type_info.impls_constraints);
    } else {
      auto generic_interface_type =
          resolver.local_types().GetAs<SemIR::GenericInterfaceType>(
              interface_const_inst.type_id());
      auto specific_id =
          GetOrAddLocalSpecific(resolver, interface.specific_id, specific_data);
      impls_constraints.push_back(
          {generic_interface_type.interface_id, specific_id});
    }
  }
  llvm::SmallVector<SemIR::FacetTypeInfo::RewriteConstraint>
      rewrite_constraints;
  for (auto rewrite : facet_type_info.rewrite_constraints) {
    rewrite_constraints.push_back(
        {.lhs_const_id = GetLocalConstantId(resolver, rewrite.lhs_const_id),
         .rhs_const_id = GetLocalConstantId(resolver, rewrite.rhs_const_id)});
  }
  // TODO: Also process the other requirements.
  SemIR::FacetTypeId facet_type_id =
      resolver.local_facet_types().Add(SemIR::FacetTypeInfo{
          .impls_constraints = impls_constraints,
          .rewrite_constraints = rewrite_constraints,
          .other_requirements = facet_type_info.other_requirements});
  return ResolveAs<SemIR::FacetType>(
      resolver,
      {.type_id = SemIR::TypeId::TypeType, .facet_type_id = facet_type_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::FacetValue inst) -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  auto type_inst_id = GetLocalConstantInstId(resolver, inst.type_inst_id);
  auto witness_inst_id = GetLocalConstantInstId(resolver, inst.witness_inst_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::FacetValue>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .type_inst_id = type_inst_id,
       .witness_inst_id = witness_inst_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::InterfaceWitness inst) -> ResolveResult {
  auto elements = GetLocalInstBlockContents(resolver, inst.elements_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  auto elements_id =
      GetLocalCanonicalInstBlockId(resolver, inst.elements_id, elements);
  return ResolveAs<SemIR::InterfaceWitness>(
      resolver, {.type_id = resolver.local_context().GetBuiltinType(
                     SemIR::BuiltinInstKind::WitnessType),
                 .elements_id = elements_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::InterfaceWitnessAccess inst)
    -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  auto witness_id = GetLocalConstantInstId(resolver, inst.witness_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::InterfaceWitnessAccess>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .witness_id = witness_id,
       .index = inst.index});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::IntValue inst) -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  // We can directly reuse the value IDs across file IRs. Otherwise, we need
  // to add a new canonical int in this IR.
  auto int_id = inst.int_id.is_value()
                    ? inst.int_id
                    : resolver.local_ints().AddSigned(
                          resolver.import_ints().Get(inst.int_id));

  return ResolveAs<SemIR::IntValue>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .int_id = int_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::IntType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto bit_width_id = GetLocalConstantInstId(resolver, inst.bit_width_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::IntType>(resolver,
                                   {.type_id = SemIR::TypeId::TypeType,
                                    .int_kind = inst.int_kind,
                                    .bit_width_id = bit_width_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::PointerType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto pointee_const_id = GetLocalConstantId(resolver, inst.pointee_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  auto pointee_type_id =
      resolver.local_context().GetTypeIdForTypeConstant(pointee_const_id);
  return ResolveAs<SemIR::PointerType>(
      resolver,
      {.type_id = SemIR::TypeId::TypeType, .pointee_id = pointee_type_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::SpecificFunction inst) -> ResolveResult {
  auto type_const_id = GetLocalConstantId(resolver, inst.type_id);
  auto callee_id = GetLocalConstantInstId(resolver, inst.callee_id);
  auto specific_data = GetLocalSpecificData(resolver, inst.specific_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  auto type_id =
      resolver.local_context().GetTypeIdForTypeConstant(type_const_id);
  auto specific_id =
      GetOrAddLocalSpecific(resolver, inst.specific_id, specific_data);
  return ResolveAs<SemIR::SpecificFunction>(
      resolver,
      {.type_id = type_id, .callee_id = callee_id, .specific_id = specific_id});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::StructType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto orig_fields = resolver.import_struct_type_fields().Get(inst.fields_id);
  llvm::SmallVector<SemIR::ConstantId> field_const_ids;
  field_const_ids.reserve(orig_fields.size());
  for (auto field : orig_fields) {
    field_const_ids.push_back(GetLocalConstantId(resolver, field.type_id));
  }
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  // Prepare a vector of fields for GetStructType.
  llvm::SmallVector<SemIR::StructTypeField> new_fields;
  new_fields.reserve(orig_fields.size());
  for (auto [orig_field, field_const_id] :
       llvm::zip(orig_fields, field_const_ids)) {
    auto name_id = GetLocalNameId(resolver, orig_field.name_id);
    auto field_type_id =
        resolver.local_context().GetTypeIdForTypeConstant(field_const_id);
    new_fields.push_back({.name_id = name_id, .type_id = field_type_id});
  }

  return ResolveAs<SemIR::StructType>(
      resolver, {.type_id = SemIR::TypeId::TypeType,
                 .fields_id = resolver.local_struct_type_fields().AddCanonical(
                     new_fields)});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::StructValue inst) -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  auto elems = GetLocalInstBlockContents(resolver, inst.elements_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::StructValue>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .elements_id =
           GetLocalCanonicalInstBlockId(resolver, inst.elements_id, elems)});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::TupleType inst) -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);

  auto orig_elem_type_ids = resolver.import_type_blocks().Get(inst.elements_id);
  llvm::SmallVector<SemIR::ConstantId> elem_const_ids;
  elem_const_ids.reserve(orig_elem_type_ids.size());
  for (auto elem_type_id : orig_elem_type_ids) {
    elem_const_ids.push_back(GetLocalConstantId(resolver, elem_type_id));
  }
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  // Prepare a vector of the tuple types for GetTupleType.
  llvm::SmallVector<SemIR::TypeId> elem_type_ids;
  elem_type_ids.reserve(orig_elem_type_ids.size());
  for (auto elem_const_id : elem_const_ids) {
    elem_type_ids.push_back(
        resolver.local_context().GetTypeIdForTypeConstant(elem_const_id));
  }

  return ResolveResult::Done(resolver.local_types().GetConstantId(
      resolver.local_context().GetTupleType(elem_type_ids)));
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::TupleValue inst) -> ResolveResult {
  auto type_id = GetLocalConstantId(resolver, inst.type_id);
  auto elems = GetLocalInstBlockContents(resolver, inst.elements_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::TupleValue>(
      resolver,
      {.type_id = resolver.local_context().GetTypeIdForTypeConstant(type_id),
       .elements_id =
           GetLocalCanonicalInstBlockId(resolver, inst.elements_id, elems)});
}

static auto TryResolveTypedInst(ImportRefResolver& resolver,
                                SemIR::UnboundElementType inst)
    -> ResolveResult {
  CARBON_CHECK(inst.type_id == SemIR::TypeId::TypeType);
  auto class_const_id = GetLocalConstantId(resolver, inst.class_type_id);
  auto elem_const_id = GetLocalConstantId(resolver, inst.element_type_id);
  if (resolver.HasNewWork()) {
    return ResolveResult::Retry();
  }

  return ResolveAs<SemIR::UnboundElementType>(
      resolver,
      {.type_id = SemIR::TypeId::TypeType,
       .class_type_id =
           resolver.local_context().GetTypeIdForTypeConstant(class_const_id),
       .element_type_id =
           resolver.local_context().GetTypeIdForTypeConstant(elem_const_id)});
}

// Tries to resolve the InstId, returning a canonical constant when ready, or
// Invalid if more has been added to the stack. This is the same as
// TryResolveInst, except that it may resolve symbolic constants as canonical
// constants instead of as constants associated with a particular generic.
static auto TryResolveInstCanonical(ImportRefResolver& resolver,
                                    SemIR::InstId inst_id,
                                    SemIR::ConstantId const_id)
    -> ResolveResult {
  if (inst_id.is_builtin()) {
    CARBON_CHECK(!const_id.is_valid());
    // Constants for builtins can be directly copied.
    return ResolveResult::Done(resolver.local_constant_values().Get(inst_id));
  }

  auto untyped_inst = resolver.import_insts().Get(inst_id);
  CARBON_KIND_SWITCH(untyped_inst) {
    case CARBON_KIND(SemIR::AdaptDecl inst): {
      return TryResolveTypedInst(resolver, inst, inst_id);
    }
    case CARBON_KIND(SemIR::AssociatedEntity inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::AssociatedEntityType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::BaseDecl inst): {
      return TryResolveTypedInst(resolver, inst, inst_id);
    }
    case CARBON_KIND(SemIR::BindAlias inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case SemIR::BindName::Kind: {
      // TODO: Should we be resolving BindNames at all?
      return ResolveResult::Done(SemIR::ConstantId::NotConstant);
    }
    case CARBON_KIND(SemIR::BindSymbolicName inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::ClassDecl inst): {
      return TryResolveTypedInst(resolver, inst, const_id);
    }
    case CARBON_KIND(SemIR::ClassType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::CompleteTypeWitness inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::ConstType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::ExportDecl inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::FacetAccessType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::FacetAccessWitness inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::FacetType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::FacetValue inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::FieldDecl inst): {
      return TryResolveTypedInst(resolver, inst, inst_id);
    }
    case CARBON_KIND(SemIR::FunctionDecl inst): {
      return TryResolveTypedInst(resolver, inst, const_id);
    }
    case CARBON_KIND(SemIR::FunctionType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::GenericClassType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::GenericInterfaceType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::ImplDecl inst): {
      return TryResolveTypedInst(resolver, inst, const_id);
    }
    case CARBON_KIND(SemIR::ImportRefLoaded inst): {
      return TryResolveTypedInst(resolver, inst, inst_id);
    }
    case CARBON_KIND(SemIR::InterfaceDecl inst): {
      return TryResolveTypedInst(resolver, inst, const_id);
    }
    case CARBON_KIND(SemIR::InterfaceWitness inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::InterfaceWitnessAccess inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::IntValue inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::IntType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::PointerType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::SpecificFunction inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::SymbolicBindingPattern inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::StructType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::StructValue inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::TupleType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::TupleValue inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    case CARBON_KIND(SemIR::UnboundElementType inst): {
      return TryResolveTypedInst(resolver, inst);
    }
    default: {
      // This instruction might have a constant value of a different kind.
      auto constant_inst_id =
          resolver.import_constant_values().GetConstantInstId(inst_id);
      if (constant_inst_id == inst_id) {
        resolver.local_context().TODO(
            SemIR::LocId(AddImportIRInst(resolver, inst_id)),
            llvm::formatv("TryResolveInst on {0}", untyped_inst.kind()).str());
        return {.const_id = SemIR::ConstantId::Error};
      }
      // Try to resolve the constant value instead. Note that this can only
      // retry once.
      CARBON_DCHECK(resolver.import_constant_values().GetConstantInstId(
                        constant_inst_id) == constant_inst_id,
                    "Constant value of constant instruction should refer to "
                    "the same instruction");
      return TryResolveInstCanonical(resolver, constant_inst_id, const_id);
    }
  }
}

// Tries to resolve the InstId, returning a constant when ready, or Invalid if
// more has been added to the stack. A similar API is followed for all
// following TryResolveTypedInst helper functions.
//
// `const_id` is Invalid unless we've tried to resolve this instruction
// before, in which case it's the previous result.
//
// TODO: Error is returned when support is missing, but that should go away.
static auto TryResolveInst(ImportRefResolver& resolver, SemIR::InstId inst_id,
                           SemIR::ConstantId const_id) -> ResolveResult {
  auto inst_const_id = resolver.import_constant_values().Get(inst_id);
  if (!inst_const_id.is_valid() || !inst_const_id.is_symbolic()) {
    return TryResolveInstCanonical(resolver, inst_id, const_id);
  }

  // Try to import the generic. This might add new work.
  const auto& symbolic_const =
      resolver.import_constant_values().GetSymbolicConstant(inst_const_id);
  auto generic_const_id =
      GetLocalConstantId(resolver, symbolic_const.generic_id);

  auto inner_const_id = SemIR::ConstantId::Invalid;
  if (const_id.is_valid()) {
    // For the third phase, extract the constant value that
    // TryResolveInstCanonical produced previously.
    inner_const_id = resolver.local_constant_values().Get(
        resolver.local_constant_values().GetSymbolicConstant(const_id).inst_id);
  }

  // Import the constant and rebuild the symbolic constant data.
  auto result = TryResolveInstCanonical(resolver, inst_id, inner_const_id);
  if (!result.const_id.is_valid()) {
    // First phase: TryResolveInstCanoncial needs a retry.
    return result;
  }

  if (!const_id.is_valid()) {
    // Second phase: we have created an abstract constant. Create a
    // corresponding generic constant.
    if (symbolic_const.generic_id.is_valid()) {
      result.const_id = resolver.local_constant_values().AddSymbolicConstant(
          {.inst_id =
               resolver.local_constant_values().GetInstId(result.const_id),
           .generic_id = GetLocalGenericId(resolver, generic_const_id),
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

// Resolves and returns the local contents for an imported instruction block
// of constant instructions.
static auto ResolveLocalInstBlockContents(ImportRefResolver& resolver,
                                          SemIR::InstBlockId import_block_id)
    -> llvm::SmallVector<SemIR::InstId> {
  auto import_block = resolver.import_inst_blocks().Get(import_block_id);

  llvm::SmallVector<SemIR::InstId> inst_ids;
  inst_ids.reserve(import_block.size());
  for (auto import_inst_id : import_block) {
    inst_ids.push_back(resolver.local_constant_values().GetInstId(
        resolver.ResolveOneInst(import_inst_id)));
  }
  return inst_ids;
}

// Resolves and returns a local eval block for a region of an imported
// generic.
static auto ResolveLocalEvalBlock(ImportRefResolver& resolver,
                                  const SemIR::Generic& import_generic,
                                  SemIR::GenericId generic_id,
                                  SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId {
  auto import_block_id = import_generic.GetEvalBlock(region);
  if (!import_block_id.is_valid()) {
    return SemIR::InstBlockId::Invalid;
  }

  auto inst_ids = ResolveLocalInstBlockContents(resolver, import_block_id);
  return RebuildGenericEvalBlock(resolver.local_context(), generic_id, region,
                                 inst_ids);
}

// Fills in the remaining information in a partially-imported generic.
static auto FinishPendingGeneric(ImportRefResolver& resolver,
                                 ImportContext::PendingGeneric pending)
    -> void {
  const auto& import_generic =
      resolver.import_generics().Get(pending.import_id);

  // Don't store the local generic between calls: the generics list can be
  // reallocated by ResolveLocalEvalBlock importing more specifics.

  auto decl_block_id =
      ResolveLocalEvalBlock(resolver, import_generic, pending.local_id,
                            SemIR::GenericInstIndex::Region::Declaration);
  resolver.local_generics().Get(pending.local_id).decl_block_id = decl_block_id;

  auto self_specific_id =
      MakeSelfSpecific(resolver.local_context(), pending.local_id);
  resolver.local_generics().Get(pending.local_id).self_specific_id =
      self_specific_id;
  resolver.AddPendingSpecific({.import_id = import_generic.self_specific_id,
                               .local_id = self_specific_id});

  auto definition_block_id =
      ResolveLocalEvalBlock(resolver, import_generic, pending.local_id,
                            SemIR::GenericInstIndex::Region::Definition);
  resolver.local_generics().Get(pending.local_id).definition_block_id =
      definition_block_id;
}

// Resolves and returns a local inst block of constant instructions
// corresponding to an imported inst block.
static auto ResolveLocalInstBlock(ImportRefResolver& resolver,
                                  SemIR::InstBlockId import_block_id)
    -> SemIR::InstBlockId {
  if (!import_block_id.is_valid()) {
    return SemIR::InstBlockId::Invalid;
  }

  auto inst_ids = ResolveLocalInstBlockContents(resolver, import_block_id);
  return resolver.local_inst_blocks().Add(inst_ids);
}

// Fills in the remaining information in a partially-imported specific.
static auto FinishPendingSpecific(ImportRefResolver& resolver,
                                  ImportContext::PendingSpecific pending)
    -> void {
  const auto& import_specific =
      resolver.import_specifics().Get(pending.import_id);

  // Don't store the local specific between calls: the specifics list can be
  // reallocated by ResolveLocalInstBlock importing more specifics.

  if (!resolver.local_specifics()
           .Get(pending.local_id)
           .decl_block_id.is_valid()) {
    auto decl_block_id =
        ResolveLocalInstBlock(resolver, import_specific.decl_block_id);
    resolver.local_specifics().Get(pending.local_id).decl_block_id =
        decl_block_id;
  }

  if (!resolver.local_specifics()
           .Get(pending.local_id)
           .definition_block_id.is_valid() &&
      import_specific.definition_block_id.is_valid()) {
    auto definition_block_id =
        ResolveLocalInstBlock(resolver, import_specific.definition_block_id);
    resolver.local_specifics().Get(pending.local_id).definition_block_id =
        definition_block_id;
  }
}

// Perform any work that we deferred until the end of the main Resolve loop.
auto ImportRefResolver::PerformPendingWork() -> void {
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
      FinishPendingGeneric(*this, pending_generics_[i]);
    }
    pending_generics_.clear();

    while (!pending_specifics_.empty()) {
      FinishPendingSpecific(*this, pending_specifics_.pop_back_val());
    }
  }
}

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

auto ImportImplsFromApiFile(Context& context) -> void {
  SemIR::ImportIRId import_ir_id = SemIR::ImportIRId::ApiForImpl;
  auto& import_ir = context.import_irs().Get(import_ir_id);
  if (!import_ir.sem_ir) {
    return;
  }

  for (auto impl_index : llvm::seq(import_ir.sem_ir->impls().size())) {
    SemIR::ImplId impl_id(impl_index);

    // Resolve the imported impl to a local impl ID.
    ImportImpl(context, import_ir_id, impl_id);
  }
}

auto ImportImpl(Context& context, SemIR::ImportIRId import_ir_id,
                SemIR::ImplId impl_id) -> void {
  ImportRefResolver resolver(context, import_ir_id);
  context.generic_region_stack().Push();

  resolver.Resolve(context.import_irs()
                       .Get(import_ir_id)
                       .sem_ir->impls()
                       .Get(impl_id)
                       .first_decl_id());

  context.generic_region_stack().Pop();
}

}  // namespace Carbon::Check
