// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/value_stores.h"

#include "llvm/ADT/StringSwitch.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_profile.h"

namespace Carbon::SemIR {

auto ConstantStore::GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId {
  // Compute the instruction's profile.
  ConstantNode node = {.inst = inst, .constant_id = ConstantId::NotConstant};
  llvm::FoldingSetNodeID id;
  node.Profile(id, constants_.getContext());

  // Check if we have already created this constant.
  void* insert_pos;
  if (ConstantNode* found = constants_.FindNodeOrInsertPos(id, insert_pos)) {
    CARBON_CHECK(found->constant_id.is_constant())
        << "Found non-constant in constant store for " << inst;
    CARBON_CHECK(found->constant_id.is_symbolic() == is_symbolic)
        << "Mismatch in phase for constant " << inst;
    return found->constant_id;
  }

  // Create the new inst and insert the new node.
  auto inst_id = constants_.getContext()->insts().AddInNoBlock(
      ParseNodeAndInst::Untyped(Parse::NodeId::Invalid, inst));
  auto constant_id = is_symbolic
                         ? SemIR::ConstantId::ForSymbolicConstant(inst_id)
                         : SemIR::ConstantId::ForTemplateConstant(inst_id);
  node.constant_id = constant_id;
  constants_.InsertNode(new (*allocator_) ConstantNode(node), insert_pos);

  // The constant value of any constant instruction is that instruction itself.
  constants_.getContext()->constant_values().Set(inst_id, constant_id);
  return constant_id;
}

auto ConstantStore::GetAsVector() const -> llvm::SmallVector<InstId, 0> {
  llvm::SmallVector<InstId, 0> result;
  result.reserve(constants_.size());
  for (const ConstantNode& node : constants_) {
    result.push_back(node.constant_id.inst_id());
  }
  // For stability, put the results into index order. This happens to also be
  // insertion order.
  std::sort(result.begin(), result.end(),
            [](InstId a, InstId b) { return a.index < b.index; });
  return result;
}

auto ConstantStore::ConstantNode::Profile(llvm::FoldingSetNodeID& id,
                                          File* sem_ir) -> void {
  ProfileConstant(id, *sem_ir, inst);
}

// Get the spelling to use for a special name.
static auto GetSpecialName(NameId name_id, bool for_ir) -> llvm::StringRef {
  switch (name_id.index) {
    case NameId::Invalid.index:
      return for_ir ? "" : "<invalid>";
    case NameId::SelfValue.index:
      return "self";
    case NameId::SelfType.index:
      return "Self";
    case NameId::ReturnSlot.index:
      return for_ir ? "return" : "<return slot>";
    case NameId::PackageNamespace.index:
      return "package";
    case NameId::Base.index:
      return "base";
    default:
      CARBON_FATAL() << "Unknown special name";
  }
}

auto NameStoreWrapper::GetFormatted(NameId name_id) const -> llvm::StringRef {
  // If the name is an identifier name with a keyword spelling, format it with
  // an `r#` prefix. Format any other identifier name as just the identifier.
  if (auto string_name = GetAsStringIfIdentifier(name_id)) {
    return llvm::StringSwitch<llvm::StringRef>(*string_name)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, "r#" Spelling)
#include "toolchain/lex/token_kind.def"
        .Default(*string_name);
  }
  return GetSpecialName(name_id, /*for_ir=*/false);
}

auto NameStoreWrapper::GetIRBaseName(NameId name_id) const -> llvm::StringRef {
  if (auto string_name = GetAsStringIfIdentifier(name_id)) {
    return *string_name;
  }
  return GetSpecialName(name_id, /*for_ir=*/true);
}

auto NameScope::Print(llvm::raw_ostream& out) const -> void {
  out << "{inst: " << inst_id << ", enclosing_scope: " << enclosing_scope_id
      << ", has_error: " << (has_error ? "true" : "false");

  out << ", extended_scopes: [";
  llvm::ListSeparator scope_sep;
  for (auto id : extended_scopes) {
    out << scope_sep << id;
  }
  out << "]";

  out << ", names: {";
  // Sort name keys to get stable output.
  llvm::SmallVector<NameId> keys;
  for (auto [key, _] : names) {
    keys.push_back(key);
  }
  llvm::sort(keys,
             [](NameId lhs, NameId rhs) { return lhs.index < rhs.index; });
  llvm::ListSeparator key_sep;
  for (auto key : keys) {
    out << key_sep << key << ": " << names.find(key)->second;
  }
  out << "}";

  out << "}";
}

}  // namespace Carbon::SemIR
