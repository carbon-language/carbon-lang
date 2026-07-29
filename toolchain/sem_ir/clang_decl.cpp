// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/clang_decl.h"

#include "clang/AST/DeclBase.h"
#include "clang/AST/TextNodeDumper.h"
#include "common/hashtable_key_context.h"
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

auto ClangDeclKey::ForFunctionDecl(clang::FunctionDecl* decl,
                                   ClangDeclSignatureId signature_id)
    -> ClangDeclKey {
  return ClangDeclKey(decl, signature_id, UncheckedTag());
}

auto ClangDeclKey::ForNonFunctionDecl(clang::Decl* decl) -> ClangDeclKey {
  CARBON_CHECK(!isa<clang::FunctionDecl>(decl));
  return ClangDeclKey(decl, ClangDeclSignatureId::None, UncheckedTag());
}

ClangDeclKey::ClangDeclKey(clang::Decl* decl, ClangDeclSignatureId signature_id,
                           UncheckedTag /*_*/)
    : decl(decl->getCanonicalDecl()), signature_id(signature_id) {}

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

class ClangDeclStore::KeyContext : public TranslatingKeyContext<KeyContext> {
 public:
  // A lookup key for a clang declaration.
  struct Key {
    InstId inst_id;
    SpecificId specific_id;

    friend auto operator==(const Key&, const Key&) -> bool = default;
  };

  explicit KeyContext(const ClangDeclStore* store) : store_(store) {}

  auto TranslateKey(ClangDeclId id) const -> Key {
    const auto& clang_decl = store_->Get(id);
    return {.inst_id = clang_decl.inst_id,
            .specific_id = clang_decl.specific_id};
  }

 private:
  const ClangDeclStore* store_;
};

ClangDeclStore::ClangDeclStore(CheckIRId check_ir_id) : values_(check_ir_id) {}

auto ClangDeclStore::Add(ClangDecl value) -> ClangDeclId {
  auto id = values_.Add(value);
  reverse_lookup_.Insert(
      KeyContext::Key{.inst_id = value.inst_id,
                      .specific_id = value.specific_id},
      [&] { return id; }, KeyContext(this));
  return id;
}

auto ClangDeclStore::LookupId(ClangDeclKey key) const -> ClangDeclId {
  return values_.Lookup(key);
}

auto ClangDeclStore::Lookup(InstId inst_id, SpecificId specific_id) const
    -> const ClangDecl* {
  if (auto result = reverse_lookup_.Lookup(
          KeyContext::Key{.inst_id = inst_id, .specific_id = specific_id},
          KeyContext(this))) {
    return &Get(result.key());
  }
  return nullptr;
}

auto ClangDeclStore::OutputYaml() const -> Yaml::OutputMapping {
  return values_.OutputYaml();
}

auto ClangDeclStore::CollectMemUsage(MemUsage& mem_usage,
                                     llvm::StringRef label) const -> void {
  values_.CollectMemUsage(mem_usage, label);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "reverse_lookup_"),
                    reverse_lookup_, KeyContext(this));
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
