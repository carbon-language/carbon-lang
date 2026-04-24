// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/clang_decl.h"

#include "clang/AST/DeclBase.h"
#include "clang/AST/TextNodeDumper.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"

namespace Carbon::SemIR {

auto Signature::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Normal:
      out << "normal";
      break;
    case TuplePattern:
      out << "tuple";
      break;
  }
  out << ", num_params: " << num_params;
  if (!passing_modes.empty()) {
    out << ", modes: ";
    for (auto mode : passing_modes) {
      out << (mode == PassingMode::Move ? "M" : "C");
    }
  }
  out << "}";
}

auto ClangDeclKey::Print(llvm::raw_ostream& out) const -> void {
  RawStringOstream decl_stream;
  auto policy = decl->getASTContext().getPrintingPolicy();
  policy.TerseOutput = true;
  if (isa<clang::TranslationUnitDecl>(decl)) {
    decl_stream << "<translation unit>";
  } else {
    decl->print(decl_stream, policy);
  }

  out << "{decl: \"" << FormatEscaped(decl_stream.TakeStr()) << "\"";
  if (signature_id != SignatureId::None) {
    out << ", signature_id: " << signature_id;
  }
  out << "}";
}

auto ClangDecl::Print(llvm::raw_ostream& out) const -> void {
  out << "{key: " << key << ", inst_id: " << inst_id << "}";
}

}  // namespace Carbon::SemIR
