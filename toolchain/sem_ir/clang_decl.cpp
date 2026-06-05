// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/clang_decl.h"

#include "clang/AST/DeclBase.h"
#include "clang/AST/TextNodeDumper.h"
#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "toolchain/base/canonical_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"

namespace Carbon::SemIR {

auto ClangDeclSignature::Print(llvm::raw_ostream& out) const -> void {
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

  auto print_mode = [&](PassingMode mode) {
    switch (mode) {
      case PassingMode::ByValue:
        out << "value";
        break;
      case PassingMode::ByVar:
        out << "var";
        break;
      case PassingMode::ByRef:
        out << "ref";
        break;
    }
  };

  if (!passing_modes.empty() && llvm::any_of(passing_modes, [](auto mode) {
        return mode != PassingMode::ByVar;
      })) {
    out << ", modes: [";
    llvm::ListSeparator sep;
    for (auto mode : passing_modes) {
      out << sep;
      print_mode(mode);
    }
    out << "]";
  }
  if (self_passing_mode != PassingMode::ByRef) {
    out << ", self_mode: ";
    print_mode(self_passing_mode);
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
  if (signature_id != ClangDeclSignatureId::None) {
    out << ", clang_decl_signature_id: " << signature_id;
  }
  out << "}";
}

auto ClangDecl::Print(llvm::raw_ostream& out) const -> void {
  out << "{key: " << key << ", inst_id: " << inst_id << "}";
}

ClangDeclStore::ClangDeclStore(CheckIRId check_ir_id) : values_(check_ir_id) {}

auto ClangDeclStore::Add(ClangDecl value) -> ClangDeclId {
  CARBON_CHECK(!isa<clang::VarDecl>(value.key.decl));
  auto id = values_.Add(value);
  inst_id_to_clang_decl_id_.Insert(value.inst_id, id);
  return id;
}

auto ClangDeclStore::AddVar(ClangDecl value, InstId pattern_id) -> ClangDeclId {
  CARBON_CHECK(isa<clang::VarDecl>(value.key.decl));
  auto id = values_.Add(value);
  inst_id_to_clang_decl_id_.Insert(pattern_id, id);
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

namespace Carbon {
template class CanonicalValueStore<SemIR::ClangDeclId, SemIR::ClangDeclKey,
                                   Tag<SemIR::CheckIRId>, SemIR::ClangDecl>;
template class ValueStore<SemIR::ClangDeclId, SemIR::ClangDecl,
                          Tag<SemIR::CheckIRId>>;
template class CanonicalValueStore<
    SemIR::ClangDeclSignatureId, SemIR::ClangDeclSignature,
    Tag<SemIR::CheckIRId>, SemIR::ClangDeclSignature>;
template class ValueStore<SemIR::ClangDeclSignatureId,
                          SemIR::ClangDeclSignature, Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
