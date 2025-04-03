// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/absolute_node_id.h"

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Notes an import on the diagnostic and updates cursors to point at the
// imported IR.
static auto FollowImportRef(
    llvm::SmallVector<AbsoluteNodeId>& absolute_node_ids,
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
    absolute_node_ids.push_back({.check_ir_id = cursor_ir->check_ir_id(),
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
    absolute_node_ids.push_back(
        {.check_ir_id = implicit_ir.sem_ir->check_ir_id(),
         .node_id = implicit_loc_id.node_id()});
  }

  cursor_ir = import_ir.sem_ir;
  cursor_inst_id = import_ir_inst.inst_id;
}

// Returns true if this is the final parse node location. If the location is is
// an import, follows it and returns false.
static auto HandleLocId(llvm::SmallVector<AbsoluteNodeId>& absolute_node_ids,
                        const File*& cursor_ir, InstId& cursor_inst_id,
                        LocId loc_id) -> bool {
  if (loc_id.is_import_ir_inst_id()) {
    FollowImportRef(absolute_node_ids, cursor_ir, cursor_inst_id,
                    loc_id.import_ir_inst_id());
    return false;
  } else {
    // Parse nodes always refer to the current IR.
    absolute_node_ids.push_back(
        {.check_ir_id = cursor_ir->check_ir_id(), .node_id = loc_id.node_id()});
    return true;
  }
}

// Loops through imported instructions until the actual instruction is found.
static auto GetAbsoluteNodeIdImpl(
    llvm::SmallVector<AbsoluteNodeId>& absolute_node_ids, const File* cursor_ir,
    InstId cursor_inst_id) -> void {
  while (cursor_inst_id.has_value()) {
    auto cursor_inst = cursor_ir->insts().Get(cursor_inst_id);
    if (auto bind_ref = cursor_inst.TryAs<ExportDecl>();
        bind_ref && bind_ref->value_id.has_value()) {
      cursor_inst_id = bind_ref->value_id;
      continue;
    }

    // If the parse node has a value, use it for the location.
    if (auto loc_id = cursor_ir->insts().GetLocId(cursor_inst_id);
        loc_id.has_value()) {
      if (HandleLocId(absolute_node_ids, cursor_ir, cursor_inst_id, loc_id)) {
        return;
      }
      continue;
    }

    // If a namespace has an instruction for an import, switch to looking at it.
    if (auto ns = cursor_inst.TryAs<Namespace>()) {
      if (ns->import_id.has_value()) {
        cursor_inst_id = ns->import_id;
        continue;
      }
    }
    break;
  }

  // `None` parse node but not an import; just nothing to point at.
  absolute_node_ids.push_back({.check_ir_id = cursor_ir->check_ir_id(),
                               .node_id = Parse::NodeId::None});
}

auto GetAbsoluteNodeId(const File* sem_ir, InstId inst_id)
    -> llvm::SmallVector<AbsoluteNodeId> {
  llvm::SmallVector<AbsoluteNodeId> absolute_node_ids;
  GetAbsoluteNodeIdImpl(absolute_node_ids, sem_ir, inst_id);
  return absolute_node_ids;
}

auto GetAbsoluteNodeId(const File* sem_ir, LocId loc_id)
    -> llvm::SmallVector<AbsoluteNodeId> {
  llvm::SmallVector<AbsoluteNodeId> absolute_node_ids;
  if (!loc_id.has_value()) {
    absolute_node_ids.push_back(
        {.check_ir_id = sem_ir->check_ir_id(), .node_id = Parse::NodeId::None});
    return absolute_node_ids;
  }
  const File* cursor_ir = sem_ir;
  InstId cursor_inst_id = InstId::None;
  if (HandleLocId(absolute_node_ids, cursor_ir, cursor_inst_id, loc_id)) {
    return absolute_node_ids;
  }
  CARBON_CHECK(cursor_inst_id.has_value(), "Should be set by HandleLocId");
  GetAbsoluteNodeIdImpl(absolute_node_ids, cursor_ir, cursor_inst_id);
  return absolute_node_ids;
}

}  // namespace Carbon::SemIR
