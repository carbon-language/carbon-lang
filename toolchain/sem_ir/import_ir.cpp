// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/import_ir.h"

#include <utility>

#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ImportIR::Print(llvm::raw_ostream& out) const -> void {
  out << "{decl_id: " << decl_id
      << ", is_export: " << (is_export ? "true" : "false") << "}";
}

auto ImportIRInst::Print(llvm::raw_ostream& out) const -> void {
  out << "{ir_id: " << ir_id() << ", ";
  if (ir_id() == ImportIRId::Cpp) {
    out << "clang_source_loc_id: " << clang_source_loc_id();
  } else {
    out << "inst_id: " << inst_id();
  }
  out << "}";
}

auto GetCanonicalFileAndInstId(const File* sem_ir, SemIR::InstId inst_id)
    -> std::pair<const File*, InstId> {
  while (true) {
    // Step through an imported instruction to the instruction it was imported
    // from.
    if (auto import_ir_inst_id = sem_ir->insts().GetImportSource(inst_id);
        import_ir_inst_id.has_value()) {
      auto import_ir_inst = sem_ir->import_ir_insts().Get(import_ir_inst_id);
      // TODO: For imports from C++, we return the importing instruction, which
      // isn't necessarily canonical.
      if (import_ir_inst.ir_id() != ImportIRId::Cpp) {
        sem_ir = sem_ir->import_irs().Get(import_ir_inst.ir_id()).sem_ir;
        inst_id = import_ir_inst.inst_id();
        continue;
      }
    }

    // Step through export declarations to their exported value.
    if (auto export_decl =
            sem_ir->insts().TryGetAs<SemIR::ExportDecl>(inst_id)) {
      inst_id = export_decl->value_id;
      continue;
    }

    // Reached a non-imported entity.
    break;
  }

  return std::make_pair(sem_ir, inst_id);
}

}  // namespace Carbon::SemIR
