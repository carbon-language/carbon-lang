// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_namer.h"

#include <string>
#include <utility>
#include <variant>

#include "common/ostream.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/cpp_overload_set.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/singleton_insts.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

class InstNamer::NamingContext {
 public:
  explicit NamingContext(InstNamer* inst_namer, InstNamer::ScopeId scope_id,
                         InstId inst_id);

  // Names the single instruction. Use bound names where available. Otherwise,
  // assign a backup name.
  //
  // Insts with a type_id are required to add names; other insts may
  // optionally set a name. All insts may push other insts to be named.
  auto NameInst() -> void;

 private:
  // Adds the instruction's name.
  auto AddInstName(std::string name) -> void;

  // Adds the instruction's name by `NameId`.
  auto AddInstNameId(NameId name_id, llvm::StringRef suffix = "") -> void {
    AddInstName((sem_ir().names().GetIRBaseName(name_id) + suffix).str());
  }

  // Names an `IntType` or `FloatType`.
  auto AddIntOrFloatTypeName(char type_literal_prefix, InstId bit_width_id,
                             llvm::StringRef suffix = "") -> void;

  // Names an `ImplWitnessTable` instruction.
  auto AddWitnessTableName(InstId witness_table_inst_id, std::string name)
      -> void;

  // Pushes all instructions in a block, by ID.
  auto PushBlockId(ScopeId scope_id, InstBlockId block_id) -> void {
    inst_namer_->PushBlockId(scope_id, block_id);
  }

  // Names the instruction as an entity. May push processing of the entity.
  template <typename EntityIdT>
  auto AddEntityNameAndMaybePush(EntityIdT id, llvm::StringRef suffix = "")
      -> void {
    AddInstName((inst_namer_->MaybePushEntity(id) + suffix).str());
  }

  auto sem_ir() -> const File& { return *inst_namer_->sem_ir_; }

  InstNamer* inst_namer_;
  ScopeId scope_id_;
  InstId inst_id_;
  Inst inst_;
};

InstNamer::InstNamer(const File* sem_ir, int total_ir_count)
    : sem_ir_(sem_ir), fingerprinter_(total_ir_count) {
  insts_.resize(sem_ir->insts().size(), {ScopeId::None, Namespace::Name()});
  labels_.resize(sem_ir->inst_blocks().size());
  scopes_.resize(GetScopeIdOffset(ScopeIdTypeEnum::None));
  generic_scopes_.resize(sem_ir->generics().size(), ScopeId::None);

  // We process the stack between each large block in order to reduce the
  // temporary size of the stack.
  auto process_stack = [&] {
    while (!inst_stack_.empty() || !inst_block_stack_.empty()) {
      if (inst_stack_.empty()) {
        auto [scope_id, block_id] = inst_block_stack_.pop_back_val();
        PushBlockInsts(scope_id, sem_ir_->inst_blocks().Get(block_id));
      }
      while (!inst_stack_.empty()) {
        auto [scope_id, inst_id] = inst_stack_.pop_back_val();
        NamingContext context(this, scope_id, inst_id);
        context.NameInst();
      }
    }
  };

  // Name each of the top-level scopes, in order. We use these as the roots of
  // walking the IR.

  PushBlockInsts(ScopeId::Constants, sem_ir->constants().array_ref());
  process_stack();

  PushBlockId(ScopeId::Imports, InstBlockId::Imports);
  process_stack();

  PushBlockId(ScopeId::File, sem_ir->top_inst_block_id());
  process_stack();

  // Global init won't have any other references, so we add it directly.
  if (sem_ir_->global_ctor_id().has_value()) {
    MaybePushEntity(sem_ir_->global_ctor_id());
    process_stack();
  }
}

auto InstNamer::GetScopeIdOffset(ScopeIdTypeEnum id_enum) const -> int {
  int offset = 0;

  // For each Id type, add the number of entities *above* its case; for example,
  // the offset for functions excludes the functions themselves. The fallthrough
  // handles summing to get uniqueness; order isn't special.
  switch (id_enum) {
    case ScopeIdTypeEnum::None:
      // `None` will be getting a full count of scopes.

      offset += sem_ir_->associated_constants().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<AssociatedConstantId>:

      offset += sem_ir_->classes().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<ClassId>:

      offset += sem_ir_->cpp_overload_sets().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<CppOverloadSetId>:

      offset += sem_ir_->functions().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<FunctionId>:

      offset += sem_ir_->impls().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<ImplId>:

      offset += sem_ir_->interfaces().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<InterfaceId>:

      offset += sem_ir_->named_constraints().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<NamedConstraintId>:

      offset += sem_ir_->specific_interfaces().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<SpecificInterfaceId>:

      offset += sem_ir_->vtables().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<VtableId>:

      // All type-specific scopes are offset by `FirstEntityScope`.
      offset += static_cast<int>(ScopeId::FirstEntityScope);
      return offset;

    default:
      CARBON_FATAL("Unexpected ScopeIdTypeEnum: {0}", id_enum);
  }
}

auto InstNamer::GetScopeName(ScopeId scope) const -> std::string {
  switch (scope) {
    case ScopeId::None:
      return "<no scope>";

    // These are treated as SemIR keywords.
    case ScopeId::File:
      return "file";
    case ScopeId::Imports:
      return "imports";
    case ScopeId::Constants:
      return "constants";

    // For everything else, use an @ prefix.
    default:
      return ("@" + GetScopeInfo(scope).name.GetFullName()).str();
  }
}

auto InstNamer::GetUnscopedNameFor(InstId inst_id) const -> llvm::StringRef {
  if (!inst_id.has_value()) {
    return "";
  }
  if (IsSingletonInstId(inst_id)) {
    return sem_ir_->insts().Get(inst_id).kind().ir_name();
  }
  auto index = sem_ir_->insts().GetRawIndex(inst_id);
  const auto& inst_name = insts_[index].second;
  return inst_name ? inst_name.GetFullName() : "";
}

auto InstNamer::GetNameFor(ScopeId scope_id, InstId inst_id) const
    -> std::string {
  if (!inst_id.has_value()) {
    return "invalid";
  }

  // Check for a builtin.
  if (IsSingletonInstId(inst_id)) {
    return sem_ir_->insts().Get(inst_id).kind().ir_name().str();
  }

  if (inst_id == SemIR::Namespace::PackageInstId) {
    return "package";
  }

  auto index = sem_ir_->insts().GetRawIndex(inst_id);
  const auto& [inst_scope, inst_name] = insts_[index];
  if (!inst_name) {
    // This should not happen in valid IR.
    RawStringOstream out;
    out << "<unexpected>." << inst_id;
    auto loc_id = sem_ir_->insts().GetCanonicalLocId(inst_id);
    // TODO: Consider handling other kinds.
    if (loc_id.kind() == LocId::Kind::NodeId) {
      const auto& tree = sem_ir_->parse_tree();
      auto token = tree.node_token(loc_id.node_id());
      out << ".loc" << tree.tokens().GetLineNumber(token) << "_"
          << tree.tokens().GetColumnNumber(token);
    }
    return out.TakeStr();
  }
  if (inst_scope == scope_id) {
    return ("%" + inst_name.GetFullName()).str();
  }
  return (GetScopeName(inst_scope) + ".%" + inst_name.GetFullName()).str();
}

auto InstNamer::GetUnscopedLabelFor(InstBlockId block_id) const
    -> llvm::StringRef {
  if (!block_id.has_value()) {
    return "";
  }
  const auto& label_name =
      labels_[sem_ir_->inst_blocks().GetRawIndex(block_id)].second;
  return label_name ? label_name.GetFullName() : "";
}

// Returns the IR name to use for a label, when referenced from a given scope.
auto InstNamer::GetLabelFor(ScopeId scope_id, InstBlockId block_id) const
    -> std::string {
  if (!block_id.has_value()) {
    return "!invalid";
  }

  const auto& [label_scope, label_name] =
      labels_[sem_ir_->inst_blocks().GetRawIndex(block_id)];
  if (!label_name) {
    // This should not happen in valid IR.
    RawStringOstream out;
    out << "<unexpected instblockref " << block_id << ">";
    return out.TakeStr();
  }
  if (label_scope == scope_id) {
    return ("!" + label_name.GetFullName()).str();
  }
  return (GetScopeName(label_scope) + ".!" + label_name.GetFullName()).str();
}

auto InstNamer::Namespace::Name::GetFullName() const -> llvm::StringRef {
  if (!value_) {
    return "<null name>";
  }
  llvm::StringMapEntry<NameResult>* value = value_;
  while (value->second.ambiguous && value->second.fallback) {
    value = value->second.fallback.value_;
  }
  return value->first();
}

auto InstNamer::Namespace::Name::GetBaseName() const -> llvm::StringRef {
  if (!value_) {
    return "<null name>";
  }
  return value_->first().take_front(base_name_size_);
}

auto InstNamer::Namespace::AllocateName(
    const InstNamer& inst_namer,
    std::variant<LocId, uint64_t> loc_id_or_fingerprint, std::string name)
    -> Name {
  // The best (shortest) name for this instruction so far, and the current
  // name for it.
  Name best;
  Name current;

  const size_t base_name_size = name.size();

  // Add `name` as a name for this entity.
  auto add_name = [&](bool mark_ambiguous = true) {
    auto [it, added] = allocated_.insert({name, NameResult()});
    Name new_name = Name(it, base_name_size);

    if (!added) {
      if (mark_ambiguous) {
        // This name was allocated for a different instruction. Mark it as
        // ambiguous and keep looking for a name for this instruction.
        new_name.SetAmbiguous();
      }
    } else {
      if (!best) {
        best = new_name;
      } else {
        CARBON_CHECK(current);
        current.SetFallback(new_name);
      }
      current = new_name;
    }
    return added;
  };

  // Use the given name if it's available.
  if (!name.empty()) {
    add_name();
  }

  // Append location information to try to disambiguate.
  if (auto* loc_id = std::get_if<LocId>(&loc_id_or_fingerprint)) {
    *loc_id = inst_namer.sem_ir_->insts().GetCanonicalLocId(*loc_id);
    // TODO: Consider handling other kinds.
    if (loc_id->kind() == LocId::Kind::NodeId) {
      const auto& tree = inst_namer.sem_ir_->parse_tree();
      auto token = tree.node_token(loc_id->node_id());
      llvm::raw_string_ostream(name)
          << ".loc" << tree.tokens().GetLineNumber(token);
      add_name();

      llvm::raw_string_ostream(name)
          << "_" << tree.tokens().GetColumnNumber(token);
      add_name();
    }
  } else {
    uint64_t fingerprint = std::get<uint64_t>(loc_id_or_fingerprint);
    llvm::raw_string_ostream out(name);
    out << ".";
    // Include names with 3-6 characters from the fingerprint. Then fall back to
    // sequential numbering.
    for (int n : llvm::seq(1, 7)) {
      out.write_hex((fingerprint >> (64 - 4 * n)) & 0xF);
      if (n >= 3) {
        add_name();
      }
    }
  }

  // Append numbers until we find an available name.
  name += ".";
  auto name_size_without_counter = name.size();
  for (int counter = 1;; ++counter) {
    name.resize(name_size_without_counter);
    llvm::raw_string_ostream(name) << counter;
    if (add_name(/*mark_ambiguous=*/false)) {
      return best;
    }
  }
}

auto InstNamer::AddBlockLabel(
    ScopeId scope_id, InstBlockId block_id, std::string name,
    std::variant<LocId, uint64_t> loc_id_or_fingerprint) -> void {
  if (!block_id.has_value() ||
      labels_[sem_ir_->inst_blocks().GetRawIndex(block_id)].second) {
    return;
  }

  if (auto* loc_id = std::get_if<LocId>(&loc_id_or_fingerprint);
      loc_id && !loc_id->has_value()) {
    if (const auto& block = sem_ir_->inst_blocks().Get(block_id);
        !block.empty()) {
      loc_id_or_fingerprint = LocId(block.front());
    }
  }

  labels_[sem_ir_->inst_blocks().GetRawIndex(block_id)] = {
      scope_id, GetScopeInfo(scope_id).labels.AllocateName(
                    *this, loc_id_or_fingerprint, std::move(name))};
}

// Provides names for `AddBlockLabel`.
struct BranchNames {
  // Returns names for a branching parse node, or nullopt if not a branch.
  static auto For(Parse::NodeKind node_kind) -> std::optional<BranchNames> {
    switch (node_kind) {
      case Parse::NodeKind::ForHeaderStart:
        return {{.prefix = "for", .branch = "next"}};

      case Parse::NodeKind::ForHeader:
        return {{.prefix = "for", .branch_if = "body", .branch = "done"}};

      case Parse::NodeKind::IfExprIf:
        return {{.prefix = "if.expr",
                 .branch_if = "then",
                 .branch = "else",
                 .branch_with_arg = "result"}};

      case Parse::NodeKind::IfCondition:
        return {{.prefix = "if", .branch_if = "then", .branch = "else"}};

      case Parse::NodeKind::IfStatement:
        return {{.prefix = "if", .branch = "done"}};

      case Parse::NodeKind::ShortCircuitOperandAnd:
        return {
            {.prefix = "and", .branch_if = "rhs", .branch_with_arg = "result"}};
      case Parse::NodeKind::ShortCircuitOperandOr:
        return {
            {.prefix = "or", .branch_if = "rhs", .branch_with_arg = "result"}};

      case Parse::NodeKind::WhileConditionStart:
        return {{.prefix = "while", .branch = "cond"}};

      case Parse::NodeKind::WhileCondition:
        return {{.prefix = "while", .branch_if = "body", .branch = "done"}};

      default:
        return std::nullopt;
    }
  }

  // Returns the provided suffix for the instruction kind, or an empty string.
  auto GetSuffix(InstKind inst_kind) -> llvm::StringLiteral {
    switch (inst_kind) {
      case BranchIf::Kind:
        return branch_if;
      case Branch::Kind:
        return branch;
      case BranchWithArg::Kind:
        return branch_with_arg;
      default:
        return "";
    }
  }

  // The kind of branch, based on the node kind.
  llvm::StringLiteral prefix;

  // For labeling branch instruction kinds. Only expected kinds need a value;
  // the empty string is for unexpected kinds.
  llvm::StringLiteral branch_if = "";
  llvm::StringLiteral branch = "";
  llvm::StringLiteral branch_with_arg = "";
};

// Finds and adds a suitable block label for the given SemIR instruction that
// represents some kind of branch.
auto InstNamer::AddBlockLabel(ScopeId scope_id, LocId loc_id, AnyBranch branch)
    -> void {
  std::string label;

  loc_id = sem_ir_->insts().GetCanonicalLocId(loc_id);
  if (loc_id.node_id().has_value()) {
    if (auto names = BranchNames::For(
            sem_ir_->parse_tree().node_kind(loc_id.node_id()))) {
      if (auto suffix = names->GetSuffix(branch.kind); !suffix.empty()) {
        label = llvm::formatv("{0}.{1}", names->prefix, suffix);
      } else {
        label =
            llvm::formatv("{0}.<unexpected {1}>", names->prefix, branch.kind);
      }
    }
  }

  AddBlockLabel(scope_id, branch.target_id, label, loc_id);
}

auto InstNamer::PushBlockId(ScopeId scope_id, InstBlockId block_id) -> void {
  if (block_id.has_value()) {
    inst_block_stack_.push_back({scope_id, block_id});
  }
}

auto InstNamer::PushBlockInsts(ScopeId scope_id,
                               llvm::ArrayRef<InstId> inst_ids) -> void {
  for (auto inst_id : llvm::reverse(inst_ids)) {
    if (inst_id.has_value() && !IsSingletonInstId(inst_id)) {
      inst_stack_.push_back(std::make_pair(scope_id, inst_id));
    }
  }
}

auto InstNamer::PushGeneric(ScopeId scope_id, GenericId generic_id) -> void {
  if (!generic_id.has_value()) {
    return;
  }
  generic_scopes_[sem_ir_->generics().GetRawIndex(generic_id)] = scope_id;
  const auto& generic = sem_ir_->generics().Get(generic_id);

  // Push blocks in reverse order.
  PushBlockId(scope_id, generic.definition_block_id);
  PushBlockId(scope_id, generic.decl_block_id);
}

auto InstNamer::PushEntity(AssociatedConstantId associated_constant_id,
                           ScopeId scope_id, Scope& scope) -> void {
  const auto& assoc_const =
      sem_ir_->associated_constants().Get(associated_constant_id);
  scope.name = globals_.AllocateName(
      *this, LocId(assoc_const.decl_id),
      sem_ir_->names().GetIRBaseName(assoc_const.name_id).str());

  // Push blocks in reverse order.
  PushGeneric(scope_id, assoc_const.generic_id);
}

auto InstNamer::PushEntity(ClassId class_id, ScopeId scope_id, Scope& scope)
    -> void {
  const auto& class_info = sem_ir_->classes().Get(class_id);
  LocId class_loc(class_info.latest_decl_id());
  scope.name = globals_.AllocateName(
      *this, class_loc,
      sem_ir_->names().GetIRBaseName(class_info.name_id).str());
  AddBlockLabel(scope_id, class_info.body_block_id, "class", class_loc);
  PushGeneric(scope_id, class_info.generic_id);

  // Push blocks in reverse order.
  PushBlockId(scope_id, class_info.body_block_id);
  PushBlockId(scope_id, class_info.pattern_block_id);
}

auto InstNamer::GetNameForParentNameScope(NameScopeId name_scope_id)
    -> llvm::StringRef {
  if (!name_scope_id.has_value()) {
    return "";
  }

  auto scope_inst =
      sem_ir_->insts().Get(sem_ir_->name_scopes().Get(name_scope_id).inst_id());
  CARBON_KIND_SWITCH(scope_inst) {
    case CARBON_KIND(ClassDecl class_decl): {
      return MaybePushEntity(class_decl.class_id);
    }
    case CARBON_KIND(ImplDecl impl): {
      return MaybePushEntity(impl.impl_id);
    }
    case CARBON_KIND(InterfaceDecl interface): {
      return MaybePushEntity(interface.interface_id);
    }
    case CARBON_KIND(NamedConstraintDecl named_constraint): {
      return MaybePushEntity(named_constraint.named_constraint_id);
    }
    case SemIR::Namespace::Kind: {
      // Only prefix type scopes.
      return "";
    }
    default: {
      return "<unsupported scope>";
    }
  }
}

auto InstNamer::PushEntity(FunctionId function_id, ScopeId scope_id,
                           Scope& scope) -> void {
  const auto& fn = sem_ir_->functions().Get(function_id);
  LocId fn_loc(fn.latest_decl_id());

  auto scope_prefix = GetNameForParentNameScope(fn.parent_scope_id);

  scope.name = globals_.AllocateName(
      *this, fn_loc,
      llvm::formatv("{0}{1}{2}", scope_prefix, scope_prefix.empty() ? "" : ".",
                    sem_ir_->names().GetIRBaseName(fn.name_id)));
  if (!fn.body_block_ids.empty()) {
    AddBlockLabel(scope_id, fn.body_block_ids.front(), "entry", fn_loc);
  }

  // Push blocks in reverse order.
  PushGeneric(scope_id, fn.generic_id);
  for (auto block_id : llvm::reverse(fn.body_block_ids)) {
    PushBlockId(scope_id, block_id);
  }
  PushBlockId(scope_id, fn.pattern_block_id);
  PushBlockId(scope_id, fn.call_params_id);
}

auto InstNamer::PushEntity(CppOverloadSetId cpp_overload_set_id,
                           ScopeId /*scope_id*/, Scope& scope) -> void {
  const CppOverloadSet& overload_set =
      sem_ir_->cpp_overload_sets().Get(cpp_overload_set_id);
  uint64_t fingerprint =
      fingerprinter_.GetOrCompute(sem_ir_, cpp_overload_set_id);

  auto scope_prefix = GetNameForParentNameScope(overload_set.parent_scope_id);
  scope.name = globals_.AllocateName(
      *this, fingerprint,
      llvm::formatv("{0}{1}{2}.cpp_overload_set", scope_prefix,
                    scope_prefix.empty() ? "" : ".",
                    sem_ir_->names().GetIRBaseName(overload_set.name_id)));
}

auto InstNamer::PushEntity(ImplId impl_id, ScopeId scope_id, Scope& scope)
    -> void {
  const auto& impl = sem_ir_->impls().Get(impl_id);
  auto impl_fingerprint = fingerprinter_.GetOrCompute(sem_ir_, impl_id);

  llvm::StringRef self_name;
  auto self_const_id =
      sem_ir_->constant_values().GetConstantInstId(impl.self_id);
  auto index = sem_ir_->insts().GetRawIndex(self_const_id);
  if (IsSingletonInstId(self_const_id)) {
    self_name = sem_ir_->insts().Get(self_const_id).kind().ir_name();
  } else if (const auto& inst_name = insts_[index].second) {
    self_name = inst_name.GetBaseName();
  } else {
    self_name = "<unexpected self>";
  }

  llvm::StringRef interface_name;
  if (impl.interface.interface_id.has_value()) {
    interface_name = MaybePushEntity(impl.interface.interface_id);
  } else {
    interface_name = "<error>";
  }

  scope.name = globals_.AllocateName(
      *this, impl_fingerprint,
      llvm::formatv("{0}.as.{1}.impl", self_name, interface_name));
  AddBlockLabel(scope_id, impl.body_block_id, "impl", impl_fingerprint);

  // Push blocks in reverse order.
  PushGeneric(scope_id, impl.generic_id);
  PushBlockId(scope_id, impl.body_block_id);
  PushBlockId(scope_id, impl.pattern_block_id);
}

auto InstNamer::PushEntity(InterfaceId interface_id, ScopeId scope_id,
                           Scope& scope) -> void {
  const auto& interface = sem_ir_->interfaces().Get(interface_id);
  LocId interface_loc(interface.latest_decl_id());
  scope.name = globals_.AllocateName(
      *this, interface_loc,
      sem_ir_->names().GetIRBaseName(interface.name_id).str());
  AddBlockLabel(scope_id, interface.body_block_id, "interface", interface_loc);

  // Push blocks in reverse order.
  PushGeneric(scope_id, interface.generic_id);
  PushBlockId(scope_id, interface.body_block_id);
  PushBlockId(scope_id, interface.pattern_block_id);
}

auto InstNamer::PushEntity(NamedConstraintId named_constraint_id,
                           ScopeId scope_id, Scope& scope) -> void {
  const auto& constraint =
      sem_ir_->named_constraints().Get(named_constraint_id);
  LocId constraint_loc(constraint.latest_decl_id());
  scope.name = globals_.AllocateName(
      *this, constraint_loc,
      sem_ir_->names().GetIRBaseName(constraint.name_id).str());
  AddBlockLabel(scope_id, constraint.body_block_id, "constraint",
                constraint_loc);

  // Push blocks in reverse order.
  PushGeneric(scope_id, constraint.generic_id);
  PushBlockId(scope_id, constraint.body_block_id);
  PushBlockId(scope_id, constraint.pattern_block_id);
}

auto InstNamer::PushEntity(VtableId vtable_id, ScopeId /*scope_id*/,
                           Scope& scope) -> void {
  const auto& vtable = sem_ir_->vtables().Get(vtable_id);
  const auto& class_info = sem_ir_->classes().Get(vtable.class_id);
  LocId vtable_loc(class_info.latest_decl_id());
  scope.name = globals_.AllocateName(
      *this, vtable_loc,
      sem_ir_->names().GetIRBaseName(class_info.name_id).str() + ".vtable");
  // TODO: Add support for generic vtables here and elsewhere.
  // PushGeneric(scope_id, vtable_info.generic_id);
}

InstNamer::NamingContext::NamingContext(InstNamer* inst_namer,
                                        InstNamer::ScopeId scope_id,
                                        InstId inst_id)
    : inst_namer_(inst_namer),
      scope_id_(scope_id),
      inst_id_(inst_id),
      inst_(sem_ir().insts().Get(inst_id)) {}

auto InstNamer::NamingContext::AddInstName(std::string name) -> void {
  auto index = sem_ir().insts().GetRawIndex(inst_id_);
  ScopeId old_scope_id = inst_namer_->insts_[index].first;
  if (old_scope_id == ScopeId::None) {
    std::variant<LocId, uint64_t> loc_id_or_fingerprint = LocId::None;
    if (scope_id_ == ScopeId::Constants || scope_id_ == ScopeId::Imports) {
      loc_id_or_fingerprint =
          inst_namer_->fingerprinter_.GetOrCompute(&sem_ir(), inst_id_);
    } else {
      loc_id_or_fingerprint = LocId(inst_id_);
    }
    auto scoped_name = inst_namer_->GetScopeInfo(scope_id_).insts.AllocateName(
        *inst_namer_, loc_id_or_fingerprint, std::move(name));
    inst_namer_->insts_[index] = {scope_id_, scoped_name};
  } else {
    CARBON_CHECK(old_scope_id == scope_id_,
                 "Attempting to name inst in multiple scopes");
  }
}

auto InstNamer::NamingContext::AddIntOrFloatTypeName(char type_literal_prefix,
                                                     InstId bit_width_id,
                                                     llvm::StringRef suffix)
    -> void {
  RawStringOstream out;
  out << type_literal_prefix;
  if (auto bit_width = sem_ir().insts().TryGetAs<IntValue>(bit_width_id)) {
    out << sem_ir().ints().Get(bit_width->int_id);
  } else {
    out << "N";
  }
  out << suffix;
  AddInstName(out.TakeStr());
}

auto InstNamer::NamingContext::AddWitnessTableName(InstId witness_table_inst_id,
                                                   std::string name) -> void {
  auto witness_table =
      sem_ir().insts().GetAs<ImplWitnessTable>(witness_table_inst_id);
  if (!witness_table.impl_id.has_value()) {
    // TODO: The witness comes from a facet value. Can we get the
    // interface names from it? Store the facet value instruction in the
    // table?
    AddInstName(name);
    return;
  }
  const auto& impl = sem_ir().impls().Get(witness_table.impl_id);
  auto name_id = sem_ir().interfaces().Get(impl.interface.interface_id).name_id;

  std::string suffix = llvm::formatv(".{}", name);
  AddInstNameId(name_id, suffix);
}

auto InstNamer::NamingContext::NameInst() -> void {
  CARBON_KIND_SWITCH(inst_) {
    case AddrOf::Kind: {
      AddInstName("addr");
      return;
    }
    case ArrayType::Kind: {
      // TODO: Can we figure out the name of the type this is an array of?
      AddInstName("array_type");
      return;
    }
    case CARBON_KIND(AssociatedConstantDecl inst): {
      AddEntityNameAndMaybePush(inst.assoc_const_id);
      PushBlockId(inst_namer_->GetScopeFor(inst.assoc_const_id),
                  inst.decl_block_id);
      return;
    }
    case CARBON_KIND(AssociatedEntity inst): {
      RawStringOstream out;
      out << "assoc" << inst.index.index;
      AddInstName(out.TakeStr());
      return;
    }
    case CARBON_KIND(AssociatedEntityType inst): {
      AddEntityNameAndMaybePush(inst.interface_id, ".assoc_type");
      return;
    }
    case AliasBinding::Kind:
    case RefBinding::Kind:
    case SymbolicBinding::Kind:
    case ValueBinding::Kind:
    case ExportDecl::Kind: {
      auto inst = inst_.As<AnyBindingOrExportDecl>();
      AddInstNameId(sem_ir().entity_names().Get(inst.entity_name_id).name_id);
      return;
    }
    case RefBindingPattern::Kind:
    case SymbolicBindingPattern::Kind:
    case ValueBindingPattern::Kind: {
      auto inst = inst_.As<AnyBindingPattern>();
      auto name_id = NameId::Underscore;
      if (inst.entity_name_id.has_value()) {
        name_id = sem_ir().entity_names().Get(inst.entity_name_id).name_id;
      }
      AddInstNameId(name_id, ".patt");
      return;
    }
    case CARBON_KIND(BoolLiteral inst): {
      if (inst.value.ToBool()) {
        AddInstName("true");
      } else {
        AddInstName("false");
      }
      return;
    }
    case CARBON_KIND(BoundMethod inst): {
      auto type_id = sem_ir().insts().Get(inst.function_decl_id).type_id();
      if (auto fn_ty = sem_ir().types().TryGetAs<FunctionType>(type_id)) {
        AddEntityNameAndMaybePush(fn_ty->function_id, ".bound");
      } else {
        AddInstName("bound_method");
      }
      return;
    }
    case Branch::Kind:
    case BranchIf::Kind:
    case BranchWithArg::Kind: {
      auto branch = inst_.As<AnyBranch>();
      inst_namer_->AddBlockLabel(scope_id_, LocId(inst_id_), branch);
      return;
    }
    case CARBON_KIND(Call inst): {
      auto callee = GetCallee(sem_ir(), inst.callee_id);
      if (auto* fn = std::get_if<CalleeFunction>(&callee)) {
        AddEntityNameAndMaybePush(fn->function_id, ".call");
        return;
      }
      AddInstName("");
      return;
    }
    case CARBON_KIND(ClassDecl inst): {
      AddEntityNameAndMaybePush(inst.class_id, ".decl");
      auto class_scope_id = inst_namer_->GetScopeFor(inst.class_id);
      PushBlockId(class_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(ClassType inst): {
      if (auto literal_info = TypeLiteralInfo::ForType(sem_ir(), inst);
          literal_info.is_valid()) {
        AddInstName(literal_info.GetLiteralAsString(sem_ir()));
      } else {
        AddEntityNameAndMaybePush(inst.class_id);
      }
      return;
    }
    case CompleteTypeWitness::Kind: {
      // TODO: Can we figure out the name of the type this is a witness for?
      AddInstName("complete_type");
      return;
    }
    case CARBON_KIND(VtableDecl inst): {
      const auto& vtable = sem_ir().vtables().Get(inst.vtable_id);
      inst_namer_->MaybePushEntity(inst.vtable_id);
      if (inst_namer_->GetScopeFor(vtable.class_id) == scope_id_) {
        inst_namer_->MaybePushEntity(vtable.class_id);
        AddInstName("vtable_decl");
      } else {
        AddEntityNameAndMaybePush(vtable.class_id, ".vtable_decl");
      }
      return;
    }
    case CARBON_KIND(VtablePtr inst): {
      const auto& vtable = sem_ir().vtables().Get(inst.vtable_id);
      inst_namer_->MaybePushEntity(inst.vtable_id);
      AddEntityNameAndMaybePush(vtable.class_id, ".vtable_ptr");
      return;
    }
    case ConstType::Kind: {
      // TODO: Can we figure out the name of the type argument?
      AddInstName("const");
      return;
    }
    case CARBON_KIND(FacetAccessType inst): {
      auto name_id = SemIR::NameId::None;
      if (auto name =
              sem_ir().insts().TryGetAs<NameRef>(inst.facet_value_inst_id)) {
        name_id = name->name_id;
      } else if (auto bind = sem_ir().insts().TryGetAs<SymbolicBinding>(
                     inst.facet_value_inst_id)) {
        name_id = sem_ir().entity_names().Get(bind->entity_name_id).name_id;
      }
      if (name_id.has_value()) {
        AddInstNameId(name_id, ".as_type");
      } else {
        AddInstName("as_type");
      }
      return;
    }
    case CARBON_KIND(SymbolicBindingType inst): {
      auto bind =
          sem_ir().insts().GetAs<SymbolicBinding>(inst.facet_value_inst_id);
      auto name_id = sem_ir().entity_names().Get(bind.entity_name_id).name_id;
      if (name_id.has_value()) {
        AddInstNameId(name_id, ".binding.as_type");
      } else {
        AddInstName("binding.as_type");
      }
      return;
    }
    case CARBON_KIND(FacetType inst): {
      const auto& facet_type_info =
          sem_ir().facet_types().Get(inst.facet_type_id);
      bool has_where = facet_type_info.other_requirements ||
                       !facet_type_info.builtin_constraint_mask.empty() ||
                       !facet_type_info.self_impls_constraints.empty() ||
                       !facet_type_info.self_impls_named_constraints.empty() ||
                       !facet_type_info.rewrite_constraints.empty();
      if (facet_type_info.extend_constraints.size() == 1 &&
          facet_type_info.extend_named_constraints.empty()) {
        AddEntityNameAndMaybePush(
            facet_type_info.extend_constraints.front().interface_id,
            has_where ? "_where.type" : ".type");
      } else if (facet_type_info.extend_named_constraints.size() == 1 &&
                 facet_type_info.extend_constraints.empty()) {
        AddEntityNameAndMaybePush(
            facet_type_info.extend_named_constraints.front()
                .named_constraint_id,
            has_where ? "_where.type" : ".type");
      } else if (facet_type_info.extend_constraints.empty() &&
                 facet_type_info.extend_named_constraints.empty()) {
        AddInstName(has_where ? "type_where" : "type");
      } else {
        AddInstName("facet_type");
      }
      return;
    }
    case CARBON_KIND(FacetValue inst): {
      if (auto facet_type =
              sem_ir().types().TryGetAs<FacetType>(inst.type_id)) {
        const auto& facet_type_info =
            sem_ir().facet_types().Get(facet_type->facet_type_id);
        if (auto interface = facet_type_info.TryAsSingleInterface()) {
          AddEntityNameAndMaybePush(interface->interface_id, ".facet");
          return;
        }
      }
      AddInstName("facet_value");
      return;
    }
    case CARBON_KIND(FloatType inst): {
      AddIntOrFloatTypeName('f', inst.bit_width_id);
      return;
    }
    case FloatLiteralValue::Kind:
    case FloatValue::Kind: {
      AddInstName("float");
      return;
    }
    case CARBON_KIND(FunctionDecl inst): {
      AddEntityNameAndMaybePush(inst.function_id, ".decl");
      auto function_scope_id = inst_namer_->GetScopeFor(inst.function_id);
      PushBlockId(function_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(FunctionType inst): {
      AddEntityNameAndMaybePush(inst.function_id, ".type");
      return;
    }
    case CARBON_KIND(CppOverloadSetValue inst): {
      AddEntityNameAndMaybePush(inst.overload_set_id, ".value");
      return;
    }
    case CARBON_KIND(CppOverloadSetType inst): {
      AddEntityNameAndMaybePush(inst.overload_set_id, ".type");
      return;
    }
    case CARBON_KIND(GenericClassType inst): {
      AddEntityNameAndMaybePush(inst.class_id, ".type");
      return;
    }
    case CARBON_KIND(GenericInterfaceType inst): {
      AddEntityNameAndMaybePush(inst.interface_id, ".type");
      return;
    }
    case CARBON_KIND(GenericNamedConstraintType inst): {
      AddEntityNameAndMaybePush(inst.named_constraint_id, ".type");
      return;
    }
    case CARBON_KIND(ImplDecl inst): {
      // `impl` declarations aren't named because they aren't added to any
      // namespace, and so aren't referenced directly.
      inst_namer_->MaybePushEntity(inst.impl_id);
      PushBlockId(inst_namer_->GetScopeFor(inst.impl_id), inst.decl_block_id);
      return;
    }
    case CARBON_KIND(LookupImplWitness inst): {
      const auto& interface =
          sem_ir().specific_interfaces().Get(inst.query_specific_interface_id);
      AddEntityNameAndMaybePush(interface.interface_id, ".lookup_impl_witness");
      return;
    }
    case CARBON_KIND(ImplWitness inst): {
      AddWitnessTableName(inst.witness_table_id, "impl_witness");
      return;
    }
    case CARBON_KIND(ImplWitnessAccess inst): {
      // TODO: Include information about the impl?
      RawStringOstream out;
      out << "impl.elem" << inst.index.index;
      AddInstName(out.TakeStr());
      return;
    }
    case CARBON_KIND(ImplWitnessAccessSubstituted inst): {
      // TODO: Include information about the impl?
      RawStringOstream out;
      auto access = sem_ir().insts().GetAs<ImplWitnessAccess>(
          inst.impl_witness_access_id);
      out << "impl.elem" << access.index.index << ".subst";
      AddInstName(out.TakeStr());
      return;
    }
    case ImplWitnessAssociatedConstant::Kind: {
      AddInstName("impl_witness_assoc_constant");
      return;
    }
    case ImplWitnessTable::Kind: {
      AddWitnessTableName(inst_id_, "impl_witness_table");
      return;
    }
    case ImportCppDecl::Kind: {
      AddInstName("Cpp.import_cpp");
      return;
    }
    case CARBON_KIND(ImportDecl inst): {
      if (inst.package_id.has_value()) {
        AddInstNameId(inst.package_id, ".import");
      } else {
        AddInstName("default.import");
      }
      return;
    }
    case ImportRefUnloaded::Kind:
    case ImportRefLoaded::Kind: {
      // Build the base import name: <package>.<entity-name>
      RawStringOstream out;

      auto inst = inst_.As<AnyImportRef>();
      auto import_ir_inst =
          sem_ir().import_ir_insts().Get(inst.import_ir_inst_id);
      const auto& import_ir =
          *sem_ir().import_irs().Get(import_ir_inst.ir_id()).sem_ir;
      auto package_id = import_ir.package_id();
      if (auto ident_id = package_id.AsIdentifierId(); ident_id.has_value()) {
        out << import_ir.identifiers().Get(ident_id);
      } else {
        out << package_id.AsSpecialName();
      }
      out << ".";

      // Add entity name if available.
      if (inst.entity_name_id.has_value()) {
        auto name_id = sem_ir().entity_names().Get(inst.entity_name_id).name_id;
        out << sem_ir().names().GetIRBaseName(name_id);
      } else {
        out << "import_ref";
      }

      AddInstName(out.TakeStr());

      // When building import refs, we frequently add instructions without
      // a block. Constants that refer to them need to be separately
      // named.
      auto const_id = sem_ir().constant_values().Get(inst_id_);
      if (const_id.has_value() && const_id.is_concrete()) {
        auto const_inst_id = sem_ir().constant_values().GetInstId(const_id);
        auto index = sem_ir().insts().GetRawIndex(const_inst_id);
        if (!inst_namer_->insts_[index].second) {
          inst_namer_->PushBlockInsts(ScopeId::Imports, const_inst_id);
        }
      }
      return;
    }
    case CARBON_KIND(InstValue inst): {
      inst_namer_->PushBlockInsts(scope_id_, inst.inst_id);
      AddInstName(
          ("inst." + sem_ir().insts().Get(inst.inst_id).kind().ir_name())
              .str());
      return;
    }
    case CARBON_KIND(InterfaceDecl inst): {
      AddEntityNameAndMaybePush(inst.interface_id, ".decl");
      auto interface_scope_id = inst_namer_->GetScopeFor(inst.interface_id);
      PushBlockId(interface_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(IntType inst): {
      AddIntOrFloatTypeName(inst.int_kind == IntKind::Signed ? 'i' : 'u',
                            inst.bit_width_id, ".builtin");
      return;
    }
    case CARBON_KIND(IntValue inst): {
      RawStringOstream out;
      out << "int_" << sem_ir().ints().Get(inst.int_id);
      AddInstName(out.TakeStr());
      return;
    }
    case CARBON_KIND(NameBindingDecl inst): {
      PushBlockId(scope_id_, inst.pattern_block_id);
      return;
    }
    case CARBON_KIND(NamedConstraintDecl inst): {
      AddEntityNameAndMaybePush(inst.named_constraint_id, ".decl");
      auto interface_scope_id =
          inst_namer_->GetScopeFor(inst.named_constraint_id);
      PushBlockId(interface_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(NameRef inst): {
      AddInstNameId(inst.name_id, ".ref");
      return;
    }
    // The namespace is specified here due to the name conflict.
    case CARBON_KIND(SemIR::Namespace inst): {
      AddInstNameId(sem_ir().name_scopes().Get(inst.name_scope_id).name_id());
      return;
    }
    case OutParam::Kind:
    case RefParam::Kind:
    case ValueParam::Kind: {
      AddInstNameId(inst_.As<AnyParam>().pretty_name_id, ".param");
      return;
    }
    case OutParamPattern::Kind:
    case RefParamPattern::Kind:
    case ValueParamPattern::Kind: {
      AddInstNameId(GetPrettyNameFromPatternId(sem_ir(), inst_id_),
                    ".param_patt");
      return;
    }
    case PatternType::Kind: {
      AddInstName("pattern_type");
      return;
    }
    case PointerType::Kind: {
      AddInstName("ptr");
      return;
    }
    case RequireCompleteType::Kind: {
      AddInstName("require_complete");
      return;
    }
    case ReturnSlotPattern::Kind: {
      AddInstNameId(NameId::ReturnSlot, ".patt");
      return;
    }
    case CARBON_KIND(SpecificFunction inst): {
      auto type_id = sem_ir().insts().Get(inst.callee_id).type_id();
      if (auto fn_ty = sem_ir().types().TryGetAs<FunctionType>(type_id)) {
        AddEntityNameAndMaybePush(fn_ty->function_id, ".specific_fn");
      } else {
        AddInstName("specific_fn");
      }
      return;
    }
    case CARBON_KIND(SpecificImplFunction inst): {
      auto type_id = sem_ir().insts().Get(inst.callee_id).type_id();
      if (auto fn_ty = sem_ir().types().TryGetAs<FunctionType>(type_id)) {
        AddEntityNameAndMaybePush(fn_ty->function_id, ".specific_impl_fn");
      } else {
        AddInstName("specific_impl_fn");
      }
      return;
    }
    case ReturnSlot::Kind: {
      AddInstNameId(NameId::ReturnSlot);
      return;
    }
    case CARBON_KIND(SpliceBlock inst): {
      PushBlockId(scope_id_, inst.block_id);
      AddInstName("");
      return;
    }
    case StringLiteral::Kind: {
      AddInstName("str");
      return;
    }
    case CARBON_KIND(StructValue inst): {
      if (auto fn_ty = sem_ir().types().TryGetAs<FunctionType>(inst.type_id)) {
        AddEntityNameAndMaybePush(fn_ty->function_id);
      } else if (auto class_ty =
                     sem_ir().types().TryGetAs<ClassType>(inst.type_id)) {
        AddEntityNameAndMaybePush(class_ty->class_id, ".val");
      } else if (auto generic_class_ty =
                     sem_ir().types().TryGetAs<GenericClassType>(
                         inst.type_id)) {
        AddEntityNameAndMaybePush(generic_class_ty->class_id, ".generic");
      } else if (auto generic_interface_ty =
                     sem_ir().types().TryGetAs<GenericInterfaceType>(
                         inst.type_id)) {
        AddInstNameId(sem_ir()
                          .interfaces()
                          .Get(generic_interface_ty->interface_id)
                          .name_id,
                      ".generic");
      } else {
        if (sem_ir().inst_blocks().Get(inst.elements_id).empty()) {
          AddInstName("empty_struct");
        } else {
          AddInstName("struct");
        }
      }
      return;
    }
    case CARBON_KIND(StructType inst): {
      const auto& fields = sem_ir().struct_type_fields().Get(inst.fields_id);
      if (fields.empty()) {
        AddInstName("empty_struct_type");
        return;
      }
      std::string name = "struct_type";
      for (auto field : fields) {
        name += ".";
        name += sem_ir().names().GetIRBaseName(field.name_id).str();
      }
      AddInstName(std::move(name));
      return;
    }
    case CARBON_KIND(TupleAccess inst): {
      RawStringOstream out;
      out << "tuple.elem" << inst.index.index;
      AddInstName(out.TakeStr());
      return;
    }
    case CARBON_KIND(TupleType inst): {
      if (inst.type_elements_id == InstBlockId::Empty) {
        AddInstName("empty_tuple.type");
      } else {
        AddInstName("tuple.type");
      }
      return;
    }
    case CARBON_KIND(TupleValue inst): {
      if (sem_ir().types().Is<ArrayType>(inst.type_id)) {
        AddInstName("array");
      } else if (inst.elements_id == InstBlockId::Empty) {
        AddInstName("empty_tuple");
      } else {
        AddInstName("tuple");
      }
      return;
    }
    case CARBON_KIND(UnboundElementType inst): {
      if (auto class_ty =
              sem_ir().insts().TryGetAs<ClassType>(inst.class_type_inst_id)) {
        AddEntityNameAndMaybePush(class_ty->class_id, ".elem");
      } else {
        AddInstName("elem_type");
      }
      return;
    }
    case VarPattern::Kind: {
      AddInstNameId(GetPrettyNameFromPatternId(sem_ir(), inst_id_),
                    ".var_patt");
      return;
    }
    case CARBON_KIND(VarStorage inst): {
      if (inst.pattern_id.has_value()) {
        AddInstNameId(GetPrettyNameFromPatternId(sem_ir(), inst.pattern_id),
                      ".var");
      } else {
        AddInstName("var");
      }
      return;
    }
    default: {
      // Sequentially number all remaining values.
      if (inst_.kind().has_type()) {
        AddInstName("");
      }
      return;
    }
  }
}

auto InstNamer::has_name(InstId inst_id) const -> bool {
  return static_cast<bool>(
      insts_[sem_ir_->insts().GetRawIndex(inst_id)].second);
}

}  // namespace Carbon::SemIR
