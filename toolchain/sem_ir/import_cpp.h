// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IMPORT_CPP_H_
#define CARBON_TOOLCHAIN_SEM_IR_IMPORT_CPP_H_

#include "common/raw_string_ostream.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Per `import Cpp` data.
struct ImportCpp : Printable<ImportCpp> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{node_id: " << node_id << ", library_id: " << library_id << "}";
  }

  Parse::ImportDeclId node_id;
  StringLiteralValueId library_id;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_IMPORT_CPP_H_
