// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/clang_decl.h"

#include "clang/AST/DeclBase.h"
#include "clang/AST/TextNodeDumper.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"

namespace Carbon::SemIR {

auto ClangDeclKey::Print(llvm::raw_ostream& out) const -> void {
  RawStringOstream decl_stream;
  auto policy = decl->getASTContext().getPrintingPolicy();
  policy.TerseOutput = true;
  if (isa<clang::TranslationUnitDecl>(decl)) {
    decl_stream << "<translation unit>";
  } else {
    decl->print(decl_stream, policy);
  }

  if (params.num_params != -1) {
    out << "{decl: \"" << FormatEscaped(decl_stream.TakeStr()) << "\", kind: ";
    switch (params.kind) {
      case ClangDeclKey::FuncParams::Normal:
        out << "normal";
        break;
      case ClangDeclKey::FuncParams::TuplePattern:
        out << "tuple";
        break;
      case ClangDeclKey::FuncParams::EmptyStructPattern:
        out << "struct";
        break;
    }
    out << ", num_params: " << params.num_params << "}";
  } else {
    out << "\"" << FormatEscaped(decl_stream.TakeStr()) << "\"";
  }
}

auto ClangDecl::Print(llvm::raw_ostream& out) const -> void {
  out << "{key: " << key << ", inst_id: " << inst_id << "}";
}

}  // namespace Carbon::SemIR
