// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/sem_ir/dump.h"

#include "common/ostream.h"
#include "toolchain/sem_ir/stringify_type.h"

namespace Carbon::SemIR {

auto DumpNoNewline(const File& file, InstId inst_id) -> void {
  llvm::errs() << inst_id;
  if (inst_id.is_valid()) {
    llvm::errs() << ": " << file.insts().Get(inst_id);
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, ConstantId const_id) -> void {
  llvm::errs() << const_id;
  if (const_id.is_symbolic()) {
    llvm::errs() << ": "
                 << file.constant_values().GetSymbolicConstant(const_id);
  } else if (const_id.is_valid()) {
    llvm::errs() << ": "
                 << file.insts().Get(
                        file.constant_values().GetInstId(const_id));
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, InstId inst_id) -> void {
  DumpNoNewline(file, inst_id);
  llvm::errs() << '\n';
  if (inst_id.is_valid()) {
    Inst inst = file.insts().Get(inst_id);
    if (inst.type_id().is_valid()) {
      llvm::errs() << "  - type ";
      Dump(file, inst.type_id());
    }
    ConstantId const_id = file.constant_values().Get(inst_id);
    if (const_id.is_valid()) {
      InstId const_inst_id = file.constant_values().GetInstId(const_id);
      llvm::errs() << "  - value ";
      if (const_inst_id == inst_id) {
        llvm::errs() << const_id << '\n';
      } else {
        Dump(file, const_id);
      }
    }
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameId name_id) -> void {
  llvm::errs() << name_id;
  if (name_id.is_valid()) {
    llvm::errs() << ": `" << file.names().GetFormatted(name_id) << "`";
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, TypeId type_id) -> void {
  llvm::errs() << type_id;
  if (type_id.is_valid()) {
    InstId inst_id = file.constant_values().GetInstId(type_id.AsConstantId());
    llvm::errs() << ": " << StringifyTypeExpr(file, inst_id) << "; "
                 << file.insts().Get(inst_id);
  }
  llvm::errs() << '\n';
}

}  // namespace Carbon::SemIR

#endif  // NDEBUG
