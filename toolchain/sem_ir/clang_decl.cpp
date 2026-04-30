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

  if (signature.num_params != -1) {
    out << "{decl: \"" << FormatEscaped(decl_stream.TakeStr()) << "\", kind: ";
    switch (signature.kind) {
      case ClangDeclKey::Signature::Normal:
        out << "normal";
        break;
      case ClangDeclKey::Signature::TuplePattern:
        out << "tuple";
        break;
    }
    out << ", num_params: " << signature.num_params << "}";
  } else {
    out << "\"" << FormatEscaped(decl_stream.TakeStr()) << "\"";
  }
}

auto ClangDecl::Print(llvm::raw_ostream& out) const -> void {
  out << "{key: " << key << ", inst_id: " << inst_id << "}";
}

ClangDeclStore::ClangDeclStore(CheckIRId check_ir_id) : values_(check_ir_id) {}

auto ClangDeclStore::Add(ClangDecl value) -> ClangDeclId {
  auto id = values_.Add(value);
  inst_id_to_clang_decl_id_.Insert(value.inst_id, id);
  return id;
}

auto ClangDeclStore::Lookup(ClangDeclKey key) const -> ClangDeclId {
  return values_.Lookup(key);
}

auto ClangDeclStore::Lookup(InstId inst_id) const -> ClangDeclId {
  if (auto result = inst_id_to_clang_decl_id_.Lookup(inst_id)) {
    return result.value();
  }
  return ClangDeclId::None;
}

auto ClangDeclStore::OutputYaml() const -> Yaml::OutputMapping {
  return values_.OutputYaml();
}

auto ClangDeclStore::CollectMemUsage(MemUsage& mem_usage,
                                     llvm::StringRef label) const -> void {
  values_.CollectMemUsage(mem_usage, label);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "inst_id_to_clang_decl_id_"),
                    inst_id_to_clang_decl_id_);
}

}  // namespace Carbon::SemIR
