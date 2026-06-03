// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/name_scope.h"

#include <optional>
#include <utility>

#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/interface.h"
#include "toolchain/sem_ir/named_constraint.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

NameScopeStore::NameScopeStore(const File* file)
    // 1 reserved untagged id because the Package NameScope is used across
    // Files.
    : file_(file), values_(file->check_ir_id(), 1) {}

auto NameScope::Print(llvm::raw_ostream& out) const -> void {
  out << "{inst: " << inst_id_ << ", parent_scope: " << parent_scope_id_
      << ", has_error: " << (has_error_ ? "true" : "false");

  out << ", extended_scopes: [";
  llvm::ListSeparator scope_sep;
  for (auto id : extended_scopes_) {
    out << scope_sep << id;
  }
  out << "]";

  out << ", names: {";
  llvm::ListSeparator sep;
  for (auto entry : names_) {
    if (entry.result.is_poisoned()) {
      continue;
    }
    out << sep << entry.name_id << ": " << entry.result.target_inst_id();
  }
  out << "}";

  out << "}";
}

auto NameScope::AddRequired(Entry name_entry) -> void {
  CARBON_CHECK(!name_entry.result.is_poisoned(),
               "Cannot add a poisoned name: {0}.", name_entry.name_id);
  auto add_name = [&] {
    EntryId index(names_.size());
    names_.push_back(name_entry);
    return index;
  };
  auto result = name_map_.Insert(name_entry.name_id, add_name);
  if (!result.is_inserted()) {
    // A required name can overwrite poison.
    auto& name = names_[result.value().index];
    CARBON_CHECK(name.result.is_poisoned(), "Failed to add required name: {0}",
                 name_entry.name_id);
    name = name_entry;
  }
}

auto NameScope::LookupOrAdd(NameId name_id, InstId inst_id,
                            AccessKind access_kind)
    -> std::pair<bool, EntryId> {
  auto insert_result = name_map_.Insert(name_id, EntryId(names_.size()));
  if (!insert_result.is_inserted()) {
    return {false, EntryId(insert_result.value())};
  }

  names_.push_back({.name_id = name_id,
                    .result = ScopeLookupResult::MakeWrappedLookupResult(
                        inst_id, access_kind)});
  return {true, EntryId(names_.size() - 1)};
}

auto NameScope::LookupOrPoison(LocId loc_id, NameId name_id)
    -> std::optional<EntryId> {
  if (!name_id.AsIdentifierId().has_value()) {
    return Lookup(name_id);
  }

  auto insert_result = name_map_.Insert(name_id, EntryId(names_.size()));
  if (insert_result.is_inserted()) {
    names_.push_back({.name_id = name_id,
                      .result = ScopeLookupResult::MakePoisoned(loc_id)});
    return std::nullopt;
  }
  return insert_result.value();
}

auto NameScopeStore::GetInstIfValid(NameScopeId scope_id) const
    -> std::pair<InstId, std::optional<Inst>> {
  if (!scope_id.has_value()) {
    return {InstId::None, std::nullopt};
  }
  auto inst_id = Get(scope_id).inst_id();
  if (!inst_id.has_value()) {
    return {InstId::None, std::nullopt};
  }
  return {inst_id, file_->insts().Get(inst_id)};
}

auto NameScopeStore::IsPrivateWithinNamespace(NameId name_id,
                                              NameScopeId parent_scope_id) const
    -> bool {
  if (!name_id.has_value() || !parent_scope_id.has_value()) {
    return false;
  }
  const auto& scope = Get(parent_scope_id);
  if (auto entry_id = scope.Lookup(name_id)) {
    return scope.GetEntry(*entry_id).result.access_kind() ==
               AccessKind::Private &&
           InstIs<Namespace>(parent_scope_id);
  }
  return false;
}

auto NameScopeStore::GetScopeNameAndParent(NameScopeId scope_id) const
    -> std::pair<NameId, NameScopeId> {
  if (!scope_id.has_value()) {
    return {NameId::None, NameScopeId::None};
  }
  const auto& scope = Get(scope_id);
  if (scope.name_id().has_value()) {
    return {scope.name_id(), scope.parent_scope_id()};
  }

  auto [inst_id, inst] = GetInstIfValid(scope_id);
  if (inst) {
    CARBON_KIND_SWITCH(*inst) {
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        const auto& class_info = file_->classes().Get(class_decl.class_id);
        return {class_info.name_id, class_info.parent_scope_id};
      }
      case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
        const auto& interface_info =
            file_->interfaces().Get(interface_decl.interface_id);
        return {interface_info.name_id, interface_info.parent_scope_id};
      }
      case CARBON_KIND(SemIR::InterfaceWithSelfDecl interface_with_self_decl): {
        const auto& interface_info =
            file_->interfaces().Get(interface_with_self_decl.interface_id);
        return {interface_info.name_id, interface_info.parent_scope_id};
      }
      case CARBON_KIND(SemIR::NamedConstraintDecl named_constraint_decl): {
        const auto& constraint_info = file_->named_constraints().Get(
            named_constraint_decl.named_constraint_id);
        return {constraint_info.name_id, constraint_info.parent_scope_id};
      }
      case CARBON_KIND(
          SemIR::NamedConstraintWithSelfDecl named_constraint_with_self_decl): {
        const auto& constraint_info = file_->named_constraints().Get(
            named_constraint_with_self_decl.named_constraint_id);
        return {constraint_info.name_id, constraint_info.parent_scope_id};
      }
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        const auto& impl_info = file_->impls().Get(impl_decl.impl_id);
        return {impl_info.name_id, impl_info.parent_scope_id};
      }
      default: {
        CARBON_FATAL("Unexpected name scope inst {0}", *inst);
      }
    }
  }
  return {NameId::None, scope.parent_scope_id()};
}

auto NameScopeStore::IsPrivateToLibrary(NameId name_id,
                                        NameScopeId parent_scope_id) const
    -> bool {
  if (!name_id.has_value() || !parent_scope_id.has_value()) {
    return false;
  }
  if (IsPrivateWithinNamespace(name_id, parent_scope_id)) {
    return true;
  }
  NameScopeId scope_id = parent_scope_id;
  while (scope_id.has_value() && !IsPackage(scope_id)) {
    auto [parent_name_id, parent_parent_scope_id] =
        GetScopeNameAndParent(scope_id);
    if (IsPrivateWithinNamespace(parent_name_id, parent_parent_scope_id)) {
      return true;
    }
    scope_id = parent_parent_scope_id;
  }
  return false;
}

}  // namespace Carbon::SemIR

namespace Carbon {
template class ValueStore<SemIR::NameScopeId, SemIR::NameScope,
                          Tag<SemIR::CheckIRId>>;
}
