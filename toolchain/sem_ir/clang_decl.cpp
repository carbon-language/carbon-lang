// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/clang_decl.h"

#include "clang/AST/DeclBase.h"

namespace Carbon::SemIR {

auto ClangDeclKey::PrintFields(llvm::raw_ostream& out) const -> void {
  out << "decl: ";
  decl->print(out);
  if (params != -1) {
    out << ", params: " << params;
  }
}

auto ClangDeclKey::Print(llvm::raw_ostream& out) const -> void {
  out << "{";
  PrintFields(out);
  out << "}";
}

auto ClangDecl::Print(llvm::raw_ostream& out) const -> void {
  out << "{";
  ClangDeclKey::PrintFields(out);
  out << ", inst_id: " << inst_id << "}";
}

}  // namespace Carbon::SemIR
