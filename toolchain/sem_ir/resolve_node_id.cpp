// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/resolve_node_id.h"

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Notes an import on the diagnostic and updates cursors to point at the
// imported IR.
static auto FollowImportRef(
    llvm::SmallVector<ResolvedNodeId>& resolved_node_ids,
    const File*& cursor_ir, InstId& cursor_inst_id,
    ImportIRInstId import_ir_inst_id) -> void {
  auto import_ir_inst = cursor_ir->import_ir_insts().Get(import_ir_inst_id);
  const auto& import_ir = cursor_ir->import_irs().Get(import_ir_inst.ir_id);
  CARBON_CHECK(import_ir.decl_id.has_value(),
               "If we get `None` locations here, we may need to more "
               "thoroughly track ImportDecls.");

  auto import_loc_id = cursor_ir->insts().GetLocId(import_ir.decl_id);
  if (import_loc_id.is_node_id()) {
    // For imports in the current file, the location is simple.
    resolved_node_ids.push_back({.check_ir_id = cursor_ir->check_ir_id(),
                                 .node_id = import_loc_id.node_id()});
  } else if (import_loc_id.is_import_ir_inst_id()) {
    // For implicit imports, we need to unravel the location a little
    // further.
    auto implicit_import_ir_inst =
        cursor_ir->import_ir_insts().Get(import_loc_id.import_ir_inst_id());
    const auto& implicit_ir =
        cursor_ir->import_irs().Get(implicit_import_ir_inst.ir_id);
    auto implicit_loc_id =
        implicit_ir.sem_ir->insts().GetLocId(implicit_import_ir_inst.inst_id);
    CARBON_CHECK(implicit_loc_id.is_node_id(),
                 "Should only be one layer of implicit imports");
    resolved_node_ids.push_back(
        {.check_ir_id = implicit_ir.sem_ir->check_ir_id(),
         .node_id = implicit_loc_id.node_id()});
  }

  cursor_ir = import_ir.sem_ir;
  cursor_inst_id = import_ir_inst.inst_id;
}

// Returns true if this is the final parse node location. If the location is is
// an import, follows it and returns false.
static auto HandleLocId(llvm::SmallVector<ResolvedNodeId>& resolved_node_ids,
                        const File*& cursor_ir, InstId& cursor_inst_id,
                        LocId loc_id) -> bool {
  if (loc_id.is_import_ir_inst_id()) {
    FollowImportRef(resolved_node_ids, cursor_ir, cursor_inst_id,
                    loc_id.import_ir_inst_id());
    return false;
  } else {
    // Parse nodes always refer to the current IR.
    resolved_node_ids.push_back(
        {.check_ir_id = cursor_ir->check_ir_id(), .node_id = loc_id.node_id()});
    return true;
  }
}

// Loops through imported instructions until the actual instruction is found.
static auto ResolveNodeIdImpl(
    llvm::SmallVector<ResolvedNodeId>& resolved_node_ids, const File* cursor_ir,
    InstId cursor_inst_id) -> void {
  while (true) {
    if (cursor_inst_id.has_value()) {
      auto cursor_inst = cursor_ir->insts().Get(cursor_inst_id);
      if (auto bind_ref = cursor_inst.TryAs<ExportDecl>();
          bind_ref && bind_ref->value_id.has_value()) {
        cursor_inst_id = bind_ref->value_id;
        continue;
      }

      // If the parse node has a value, use it for the location.
      if (auto loc_id = cursor_ir->insts().GetLocId(cursor_inst_id);
          loc_id.has_value()) {
        if (HandleLocId(resolved_node_ids, cursor_ir, cursor_inst_id, loc_id)) {
          return;
        }
        continue;
      }

      // If a namespace has an instruction for an import, switch to looking at
      // it.
      if (auto ns = cursor_inst.TryAs<Namespace>()) {
        if (ns->import_id.has_value()) {
          cursor_inst_id = ns->import_id;
          continue;
        }
      }
    }

    // `None` parse node but not an import; just nothing to point at.
    resolved_node_ids.push_back({.check_ir_id = cursor_ir->check_ir_id(),
                                 .node_id = Parse::NodeId::None});
    return;
  }
}

auto ResolveNodeId(const File* sem_ir, InstId inst_id)
    -> llvm::SmallVector<ResolvedNodeId> {
  llvm::SmallVector<ResolvedNodeId> resolved_node_ids;
  ResolveNodeIdImpl(resolved_node_ids, sem_ir, inst_id);
  return resolved_node_ids;
}

auto ResolveNodeId(const File* sem_ir, LocId loc_id)
    -> llvm::SmallVector<ResolvedNodeId> {
  llvm::SmallVector<ResolvedNodeId> resolved_node_ids;
  if (!loc_id.has_value()) {
    resolved_node_ids.push_back(
        {.check_ir_id = sem_ir->check_ir_id(), .node_id = Parse::NodeId::None});
    return resolved_node_ids;
  }
  const File* cursor_ir = sem_ir;
  InstId cursor_inst_id = InstId::None;
  if (HandleLocId(resolved_node_ids, cursor_ir, cursor_inst_id, loc_id)) {
    return resolved_node_ids;
  }
  CARBON_CHECK(cursor_inst_id.has_value(), "Should be set by HandleLocId");
  ResolveNodeIdImpl(resolved_node_ids, cursor_ir, cursor_inst_id);
  return resolved_node_ids;
}

}  // namespace Carbon::SemIR
