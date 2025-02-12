// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INST_STORE_WRAPPER_H_
#define CARBON_TOOLCHAIN_CHECK_INST_STORE_WRAPPER_H_

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

class Context;

// Wraps SemIR::InstStore for check. The read-only methods are essentially the
// same, but the mutations include check-specific logic.
class InstStoreWrapper {
 public:
  explicit InstStoreWrapper(Context* context) : context_(context) {}

  // Adds an instruction to the current block, returning the produced ID.
  auto Add(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for Add with typed nodes.
  template <typename InstT, typename LocT>
  auto Add(LocT loc, InstT inst)
      -> decltype(Add(SemIR::LocIdAndInst(loc, inst))) {
    return Add(SemIR::LocIdAndInst(loc, inst));
  }

  // Pushes a parse tree node onto the node stack, storing the SemIR::Inst as
  // the result.
  template <typename InstT>
    requires(SemIR::Internal::HasNodeId<InstT>)
  auto AddAndPush(decltype(InstT::Kind)::TypedNodeId node_id, InstT inst)
      -> void {
    NodeStackPush(node_id, Add(node_id, inst));
  }

  // Returns a LocIdAndInst for an instruction with an imported location. Checks
  // that the imported location is compatible with the kind of instruction being
  // created.
  template <typename InstT>
    requires SemIR::Internal::HasNodeId<InstT>
  auto MakeImportedLocIdAndInst(SemIR::ImportIRInstId imported_loc_id,
                                InstT inst) -> SemIR::LocIdAndInst {
    if constexpr (!SemIR::Internal::HasUntypedNodeId<InstT>) {
      CheckCompatibleImportedNodeKind(imported_loc_id, InstT::Kind);
    }
    return SemIR::LocIdAndInst::UncheckedLoc(imported_loc_id, inst);
  }

  // Adds an instruction in no block, returning the produced ID. Should be used
  // rarely.
  auto AddInNoBlock(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for AddInNoBlock with typed nodes.
  template <typename InstT, typename LocT>
  auto AddInNoBlock(LocT loc, InstT inst)
      -> decltype(AddInNoBlock(SemIR::LocIdAndInst(loc, inst))) {
    return AddInNoBlock(SemIR::LocIdAndInst(loc, inst));
  }

  // If the instruction has an implicit location and a constant value, returns
  // the constant value's instruction ID. Otherwise, same as Add.
  auto GetOrAdd(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for GetOrAddInst with typed nodes.
  template <typename InstT, typename LocT>
  auto GetOrAdd(LocT loc, InstT inst)
      -> decltype(GetOrAdd(SemIR::LocIdAndInst(loc, inst))) {
    return GetOrAdd(SemIR::LocIdAndInst(loc, inst));
  }

  // Adds an instruction to the current pattern block, returning the produced
  // ID.
  // TODO: Is it possible to remove this and pattern_block_stack, now that
  // we have BeginSubpattern etc. instead?
  auto AddInPatternBlock(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Convenience for AddInPatternBlock with typed nodes.
  template <typename InstT>
    requires(SemIR::Internal::HasNodeId<InstT>)
  auto AddInPatternBlock(decltype(InstT::Kind)::TypedNodeId node_id, InstT inst)
      -> SemIR::InstId {
    return AddInPatternBlock(SemIR::LocIdAndInst(node_id, inst));
  }

  // Adds an instruction to the current block, returning the produced ID. The
  // instruction is a placeholder that is expected to be replaced by
  // `ReplaceBeforeConstantUse`.
  auto AddPlaceholder(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId;

  // Adds an instruction in no block, returning the produced ID. Should be used
  // rarely. The instruction is a placeholder that is expected to be replaced by
  // `ReplaceBeforeConstantUse`.
  auto AddPlaceholderInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
      -> SemIR::InstId;

  // Replaces the instruction at `inst_id` with `loc_id_and_inst`. The
  // instruction is required to not have been used in any constant evaluation,
  // either because it's newly created and entirely unused, or because it's only
  // used in a position that constant evaluation ignores, such as a return slot.
  auto ReplaceLocIdAndInstBeforeConstantUse(SemIR::InstId inst_id,
                                            SemIR::LocIdAndInst loc_id_and_inst)
      -> void;

  // Replaces the instruction at `inst_id` with `inst`, not affecting location.
  // The instruction is required to not have been used in any constant
  // evaluation, either because it's newly created and entirely unused, or
  // because it's only used in a position that constant evaluation ignores, such
  // as a return slot.
  auto ReplaceBeforeConstantUse(SemIR::InstId inst_id, SemIR::Inst inst)
      -> void;

  // Replaces the instruction at `inst_id` with `inst`, not affecting location.
  // The instruction is required to not change its constant value.
  auto ReplacePreservingConstantValue(SemIR::InstId inst_id, SemIR::Inst inst)
      -> void;

  // Sets only the parse node of an instruction. This is only used when setting
  // the parse node of an imported namespace. Versus ReplaceBeforeConstantUse,
  // it is safe to use after the namespace is used in constant evaluation. It's
  // exposed this way mainly so that `insts()` can remain const.
  auto SetNamespaceNodeId(SemIR::InstId inst_id, Parse::NodeId node_id) -> void;

  // Below methods are read-only and expose `InstStore` functionality directly.

  auto Get(SemIR::InstId inst_id) const -> SemIR::Inst {
    return insts().Get(inst_id);
  }
  auto GetWithLocId(SemIR::InstId inst_id) const -> SemIR::LocIdAndInst {
    return insts().GetWithLocId(inst_id);
  }
  template <typename InstT>
  auto Is(SemIR::InstId inst_id) const -> bool {
    return insts().Is<InstT>(inst_id);
  }
  template <typename InstT>
  auto GetAs(SemIR::InstId inst_id) const -> InstT {
    return insts().GetAs<InstT>(inst_id);
  }
  template <typename InstT>
  auto TryGetAs(SemIR::InstId inst_id) const -> std::optional<InstT> {
    return insts().TryGetAs<InstT>(inst_id);
  }
  template <typename InstT>
  auto TryGetAsIfValid(SemIR::InstId inst_id) const -> std::optional<InstT> {
    return insts().TryGetAsIfValid<InstT>(inst_id);
  }
  auto GetLocId(SemIR::InstId inst_id) const -> SemIR::LocId {
    return insts().GetLocId(inst_id);
  }

 private:
  // Checks that the provided imported location has a node kind that is
  // compatible with that of the given instruction.
  auto CheckCompatibleImportedNodeKind(SemIR::ImportIRInstId imported_loc_id,
                                       SemIR::InstKind kind) -> void;

  // Finish producing an instruction. Set its constant value, and register it in
  // any applicable instruction lists.
  auto Finish(SemIR::InstId inst_id, SemIR::Inst inst) -> void;

  // Pushes an InstId onto the node stack.
  auto NodeStackPush(Parse::NodeId node_id, SemIR::InstId inst_id) -> void;

  // Exposes `InstStore` to allow read-only methods in the header.
  auto insts() const -> const SemIR::InstStore&;

  Context* context_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INST_STORE_WRAPPER_H_
