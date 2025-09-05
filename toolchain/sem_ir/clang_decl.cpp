// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/clang_decl.h"

#include "clang/AST/DeclBase.h"

namespace Carbon::SemIR {

auto ClangDecl::Print(llvm::raw_ostream& out) const -> void {
  out << "{decl: ";
  decl->print(out);
  out << ", inst_id: " << inst_id << "}";
}

}  // namespace Carbon::SemIR
