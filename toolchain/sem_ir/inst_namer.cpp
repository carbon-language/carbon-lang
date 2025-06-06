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
#include "toolchain/sem_ir/builtin_function_kind.h"
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
  explicit NamingContext(InstNamer* inst_namer,
                         llvm::SmallVector<std::pair<ScopeId, InstId>>* queue,
                         InstNamer::ScopeId scope_id, InstId inst_id);

  // Names the single instruction. Use bound names where available. Otherwise,
  // assign a backup name.
  //
  // Insts with a type_id are required to add names; other insts may
  // optionally set a name. All insts may enqueue other insts to be named.
  auto NameInst() -> void;

 private:
  // Adds the instruction's name.
  auto AddInstName(std::string name) -> void;

  // Adds the instruction's name by `NameId`.
  auto AddInstNameId(NameId name_id, llvm::StringRef suffix = "") -> void {
    AddInstName((sem_ir().names().GetIRBaseName(name_id).str() + suffix).str());
  }

  // Names an `IntType` or `FloatType`.
  auto AddIntOrFloatTypeName(char type_literal_prefix, InstId bit_width_id,
                             llvm::StringRef suffix = "") -> void;

  // Names an `ImplWitnessTable` instruction.
  auto AddWitnessTableName(InstId witness_table_inst_id, std::string name)
      -> void;

  // Enqueues all instructions in a block, by ID.
  auto QueueBlockId(ScopeId scope_id, InstBlockId block_id) -> void {
    if (block_id.has_value()) {
      inst_namer_->QueueBlockInsts(*queue_, scope_id,
                                   sem_ir().inst_blocks().Get(block_id));
    }
  }

  auto sem_ir() -> const SemIR::File& { return *inst_namer_->sem_ir_; }

  InstNamer* inst_namer_;
  llvm::SmallVector<std::pair<ScopeId, InstId>>* queue_;
  ScopeId scope_id_;
  InstId inst_id_;
  Inst inst_;
};

InstNamer::InstNamer(const File* sem_ir) : sem_ir_(sem_ir) {
  insts_.resize(sem_ir->insts().size(), {ScopeId::None, Namespace::Name()});
  labels_.resize(sem_ir->inst_blocks().size());
  scopes_.resize(GetScopeIdOffset(ScopeIdTypeEnum::None));
  generic_scopes_.resize(sem_ir->generics().size(), ScopeId::None);

  // Build the constants scope.
  CollectNamesInBlock(ScopeId::Constants, sem_ir->constants().array_ref());

  // Build the imports scope.
  CollectNamesInBlock(ScopeId::Imports,
                      sem_ir->inst_blocks().Get(InstBlockId::Imports));

  // Build the file scope.
  CollectNamesInBlock(ScopeId::File, sem_ir->top_inst_block_id());

  // Build each function scope.
  for (auto [fn_id, fn] : sem_ir->functions().enumerate()) {
    auto fn_scope = GetScopeFor(fn_id);
    // TODO: Provide a location for the function for use as a
    // disambiguator.
    auto fn_loc = Parse::NodeId::None;
    GetScopeInfo(fn_scope).name = globals_.AllocateName(
        *this, fn_loc, sem_ir->names().GetIRBaseName(fn.name_id).str());
    CollectNamesInBlock(fn_scope, fn.call_params_id);
    CollectNamesInBlock(fn_scope, fn.pattern_block_id);
    if (!fn.body_block_ids.empty()) {
      AddBlockLabel(fn_scope, fn.body_block_ids.front(), "entry", fn_loc);
    }
    for (auto block_id : fn.body_block_ids) {
      CollectNamesInBlock(fn_scope, block_id);
    }
    for (auto block_id : fn.body_block_ids) {
      AddBlockLabel(fn_scope, block_id);
    }
    CollectNamesInGeneric(fn_scope, fn.generic_id);
  }

  // Build each class scope.
  for (auto [class_id, class_info] : sem_ir->classes().enumerate()) {
    auto class_scope = GetScopeFor(class_id);
    // TODO: Provide a location for the class for use as a disambiguator.
    auto class_loc = Parse::NodeId::None;
    GetScopeInfo(class_scope).name = globals_.AllocateName(
        *this, class_loc,
        sem_ir->names().GetIRBaseName(class_info.name_id).str());
    CollectNamesInBlock(class_scope, class_info.pattern_block_id);
    AddBlockLabel(class_scope, class_info.body_block_id, "class", class_loc);
    CollectNamesInBlock(class_scope, class_info.body_block_id);
    CollectNamesInGeneric(class_scope, class_info.generic_id);
  }

  // Build each interface scope.
  for (auto [interface_id, interface_info] : sem_ir->interfaces().enumerate()) {
    auto interface_scope = GetScopeFor(interface_id);
    // TODO: Provide a location for the interface for use as a disambiguator.
    auto interface_loc = Parse::NodeId::None;
    GetScopeInfo(interface_scope).name = globals_.AllocateName(
        *this, interface_loc,
        sem_ir->names().GetIRBaseName(interface_info.name_id).str());
    CollectNamesInBlock(interface_scope, interface_info.pattern_block_id);
    AddBlockLabel(interface_scope, interface_info.body_block_id, "interface",
                  interface_loc);
    CollectNamesInBlock(interface_scope, interface_info.body_block_id);
    CollectNamesInGeneric(interface_scope, interface_info.generic_id);
  }

  // Build each associated constant scope.
  for (auto [assoc_const_id, assoc_const_info] :
       sem_ir->associated_constants().enumerate()) {
    auto assoc_const_scope = GetScopeFor(assoc_const_id);
    GetScopeInfo(assoc_const_scope).name = globals_.AllocateName(
        *this, LocId(assoc_const_info.decl_id),
        sem_ir->names().GetIRBaseName(assoc_const_info.name_id).str());
    CollectNamesInGeneric(assoc_const_scope, assoc_const_info.generic_id);
  }

  // Build each impl scope.
  for (auto [impl_id, impl_info] : sem_ir->impls().enumerate()) {
    auto impl_scope = GetScopeFor(impl_id);
    auto impl_fingerprint = fingerprinter_.GetOrCompute(sem_ir_, impl_id);
    // TODO: Invent a name based on the self and constraint types.
    GetScopeInfo(impl_scope).name =
        globals_.AllocateName(*this, impl_fingerprint, "impl");
    CollectNamesInBlock(impl_scope, impl_info.pattern_block_id);
    AddBlockLabel(impl_scope, impl_info.body_block_id, "impl",
                  impl_fingerprint);
    CollectNamesInBlock(impl_scope, impl_info.body_block_id);
    CollectNamesInGeneric(impl_scope, impl_info.generic_id);
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
      offset += sem_ir_->functions().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<FunctionId>:
      offset += sem_ir_->impls().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<ImplId>:
      offset += sem_ir_->interfaces().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<InterfaceId>:
      offset += sem_ir_->specific_interfaces().size();
      [[fallthrough]];
    case ScopeIdTypeEnum::For<SpecificInterfaceId>:
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
      return ("@" + GetScopeInfo(scope).name.str()).str();
  }
}

auto InstNamer::GetUnscopedNameFor(InstId inst_id) const -> llvm::StringRef {
  if (!inst_id.has_value()) {
    return "";
  }
  const auto& inst_name = insts_[inst_id.index].second;
  return inst_name ? inst_name.str() : "";
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

  const auto& [inst_scope, inst_name] = insts_[inst_id.index];
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
    return ("%" + inst_name.str()).str();
  }
  return (GetScopeName(inst_scope) + ".%" + inst_name.str()).str();
}

auto InstNamer::GetUnscopedLabelFor(InstBlockId block_id) const
    -> llvm::StringRef {
  if (!block_id.has_value()) {
    return "";
  }
  const auto& label_name = labels_[block_id.index].second;
  return label_name ? label_name.str() : "";
}

// Returns the IR name to use for a label, when referenced from a given scope.
auto InstNamer::GetLabelFor(ScopeId scope_id, InstBlockId block_id) const
    -> std::string {
  if (!block_id.has_value()) {
    return "!invalid";
  }

  const auto& [label_scope, label_name] = labels_[block_id.index];
  if (!label_name) {
    // This should not happen in valid IR.
    RawStringOstream out;
    out << "<unexpected instblockref " << block_id << ">";
    return out.TakeStr();
  }
  if (label_scope == scope_id) {
    return ("!" + label_name.str()).str();
  }
  return (GetScopeName(label_scope) + ".!" + label_name.str()).str();
}

auto InstNamer::Namespace::Name::str() const -> llvm::StringRef {
  llvm::StringMapEntry<NameResult>* value = value_;
  CARBON_CHECK(value, "cannot print a null name");
  while (value->second.ambiguous && value->second.fallback) {
    value = value->second.fallback.value_;
  }
  return value->first();
}

auto InstNamer::Namespace::AllocateName(
    const InstNamer& inst_namer,
    std::variant<LocId, uint64_t> loc_id_or_fingerprint, std::string name)
    -> Name {
  // The best (shortest) name for this instruction so far, and the current
  // name for it.
  Name best;
  Name current;

  // Add `name` as a name for this entity.
  auto add_name = [&](bool mark_ambiguous = true) {
    auto [it, added] = allocated.insert({name, NameResult()});
    Name new_name = Name(it);

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
  if (!block_id.has_value() || labels_[block_id.index].second) {
    return;
  }

  if (auto* loc_id = std::get_if<LocId>(&loc_id_or_fingerprint);
      loc_id && !loc_id->has_value()) {
    if (const auto& block = sem_ir_->inst_blocks().Get(block_id);
        !block.empty()) {
      loc_id_or_fingerprint = LocId(block.front());
    }
  }

  labels_[block_id.index] = {
      scope_id, GetScopeInfo(scope_id).labels.AllocateName(
                    *this, loc_id_or_fingerprint, std::move(name))};
}

// Finds and adds a suitable block label for the given SemIR instruction that
// represents some kind of branch.
auto InstNamer::AddBlockLabel(ScopeId scope_id, LocId loc_id, AnyBranch branch)
    -> void {
  loc_id = sem_ir_->insts().GetCanonicalLocId(loc_id);
  if (!loc_id.node_id().has_value()) {
    AddBlockLabel(scope_id, branch.target_id, "", loc_id);
    return;
  }
  llvm::StringRef name;
  switch (sem_ir_->parse_tree().node_kind(loc_id.node_id())) {
    case Parse::NodeKind::IfExprIf:
      switch (branch.kind) {
        case BranchIf::Kind:
          name = "if.expr.then";
          break;
        case Branch::Kind:
          name = "if.expr.else";
          break;
        case BranchWithArg::Kind:
          name = "if.expr.result";
          break;
        default:
          break;
      }
      break;

    case Parse::NodeKind::IfCondition:
      switch (branch.kind) {
        case BranchIf::Kind:
          name = "if.then";
          break;
        case Branch::Kind:
          name = "if.else";
          break;
        default:
          break;
      }
      break;

    case Parse::NodeKind::IfStatement:
      name = "if.done";
      break;

    case Parse::NodeKind::ShortCircuitOperandAnd:
      name = branch.kind == BranchIf::Kind ? "and.rhs" : "and.result";
      break;
    case Parse::NodeKind::ShortCircuitOperandOr:
      name = branch.kind == BranchIf::Kind ? "or.rhs" : "or.result";
      break;

    case Parse::NodeKind::WhileConditionStart:
      name = "while.cond";
      break;

    case Parse::NodeKind::WhileCondition:
      switch (branch.kind) {
        case BranchIf::Kind:
          name = "while.body";
          break;
        case Branch::Kind:
          name = "while.done";
          break;
        default:
          break;
      }
      break;

    default:
      break;
  }

  AddBlockLabel(scope_id, branch.target_id, name.str(), loc_id);
}

auto InstNamer::CollectNamesInBlock(ScopeId scope_id, InstBlockId block_id)
    -> void {
  if (block_id.has_value()) {
    CollectNamesInBlock(scope_id, sem_ir_->inst_blocks().Get(block_id));
  }
}

auto InstNamer::CollectNamesInBlock(ScopeId top_scope_id,
                                    llvm::ArrayRef<InstId> block) -> void {
  llvm::SmallVector<std::pair<ScopeId, InstId>> queue;

  QueueBlockInsts(queue, top_scope_id, block);

  while (!queue.empty()) {
    auto [scope_id, inst_id] = queue.pop_back_val();
    NamingContext context(this, &queue, scope_id, inst_id);
    context.NameInst();
  }
}

auto InstNamer::CollectNamesInGeneric(ScopeId scope_id, GenericId generic_id)
    -> void {
  if (!generic_id.has_value()) {
    return;
  }
  generic_scopes_[generic_id.index] = scope_id;
  const auto& generic = sem_ir_->generics().Get(generic_id);
  CollectNamesInBlock(scope_id, generic.decl_block_id);
  CollectNamesInBlock(scope_id, generic.definition_block_id);
}

auto InstNamer::QueueBlockInsts(
    llvm::SmallVector<std::pair<ScopeId, InstId>>& queue, ScopeId scope_id,
    llvm::ArrayRef<InstId> inst_ids) -> void {
  for (auto inst_id : llvm::reverse(inst_ids)) {
    if (inst_id.has_value() && !IsSingletonInstId(inst_id)) {
      queue.push_back(std::make_pair(scope_id, inst_id));
    }
  }
}

InstNamer::NamingContext::NamingContext(
    InstNamer* inst_namer, llvm::SmallVector<std::pair<ScopeId, InstId>>* queue,
    InstNamer::ScopeId scope_id, InstId inst_id)
    : inst_namer_(inst_namer),
      queue_(queue),
      scope_id_(scope_id),
      inst_id_(inst_id),
      inst_(sem_ir().insts().Get(inst_id)) {}

auto InstNamer::NamingContext::AddInstName(std::string name) -> void {
  ScopeId old_scope_id = inst_namer_->insts_[inst_id_.index].first;
  if (old_scope_id == ScopeId::None) {
    std::variant<LocId, uint64_t> loc_id_or_fingerprint = LocId::None;
    if (scope_id_ == ScopeId::Constants || scope_id_ == ScopeId::Imports) {
      loc_id_or_fingerprint =
          inst_namer_->fingerprinter_.GetOrCompute(&sem_ir(), inst_id_);
    } else {
      loc_id_or_fingerprint = LocId(inst_id_);
    }
    auto scoped_name = inst_namer_->GetScopeInfo(scope_id_).insts.AllocateName(
        *inst_namer_, loc_id_or_fingerprint, name);
    inst_namer_->insts_[inst_id_.index] = {scope_id_, scoped_name};
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
      AddInstNameId(
          sem_ir().associated_constants().Get(inst.assoc_const_id).name_id);
      QueueBlockId(inst_namer_->GetScopeFor(inst.assoc_const_id),
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
      const auto& interface_info = sem_ir().interfaces().Get(inst.interface_id);
      AddInstNameId(interface_info.name_id, ".assoc_type");
      return;
    }
    case BindAlias::Kind:
    case BindName::Kind:
    case BindSymbolicName::Kind:
    case ExportDecl::Kind: {
      auto inst = inst_.As<AnyBindNameOrExportDecl>();
      AddInstNameId(sem_ir().entity_names().Get(inst.entity_name_id).name_id);
      return;
    }
    case BindingPattern::Kind:
    case SymbolicBindingPattern::Kind: {
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
        AddInstNameId(sem_ir().functions().Get(fn_ty->function_id).name_id,
                      ".bound");
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
      auto callee_function = GetCalleeFunction(sem_ir(), inst.callee_id);
      if (!callee_function.function_id.has_value()) {
        AddInstName("");
        return;
      }
      const auto& function =
          sem_ir().functions().Get(callee_function.function_id);
      // Name the call's result based on the callee.
      if (function.builtin_function_kind != BuiltinFunctionKind::None) {
        // For a builtin, use the builtin name. Otherwise, we'd typically pick
        // the name `Op` below, which is probably not very useful.
        AddInstName(function.builtin_function_kind.name().str());
        return;
      }

      AddInstNameId(function.name_id, ".call");
      return;
    }
    case CARBON_KIND(ClassDecl inst): {
      const auto& class_info = sem_ir().classes().Get(inst.class_id);
      AddInstNameId(class_info.name_id, ".decl");
      auto class_scope_id = inst_namer_->GetScopeFor(inst.class_id);
      QueueBlockId(class_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(ClassType inst): {
      if (auto literal_info = NumericTypeLiteralInfo::ForType(sem_ir(), inst);
          literal_info.is_valid()) {
        AddInstName(literal_info.GetLiteralAsString(sem_ir()));
      } else {
        AddInstNameId(sem_ir().classes().Get(inst.class_id).name_id);
      }
      return;
    }
    case CompleteTypeWitness::Kind: {
      // TODO: Can we figure out the name of the type this is a witness for?
      AddInstName("complete_type");
      return;
    }
    case ConstType::Kind: {
      // TODO: Can we figure out the name of the type argument?
      AddInstName("const");
      return;
    }
    case CARBON_KIND(FacetAccessType inst): {
      auto name_id = NameId::None;
      if (auto name =
              sem_ir().insts().TryGetAs<NameRef>(inst.facet_value_inst_id)) {
        name_id = name->name_id;
      } else if (auto symbolic = sem_ir().insts().TryGetAs<BindSymbolicName>(
                     inst.facet_value_inst_id)) {
        name_id = sem_ir().entity_names().Get(symbolic->entity_name_id).name_id;
      }

      if (name_id.has_value()) {
        AddInstNameId(name_id, ".as_type");
      } else {
        AddInstName("as_type");
      }
      return;
    }
    case CARBON_KIND(FacetType inst): {
      const auto& facet_type_info =
          sem_ir().facet_types().Get(inst.facet_type_id);
      bool has_where = facet_type_info.other_requirements ||
                       !facet_type_info.self_impls_constraints.empty() ||
                       !facet_type_info.rewrite_constraints.empty();
      if (facet_type_info.extend_constraints.size() == 1) {
        const auto& interface_info = sem_ir().interfaces().Get(
            facet_type_info.extend_constraints.front().interface_id);
        AddInstNameId(interface_info.name_id,
                      has_where ? "_where.type" : ".type");
      } else if (facet_type_info.extend_constraints.empty()) {
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
          const auto& interface_info =
              sem_ir().interfaces().Get(interface->interface_id);
          AddInstNameId(interface_info.name_id, ".facet");
          return;
        }
      }
      AddInstName("facet_value");
      return;
    }
    case FloatLiteral::Kind: {
      AddInstName("float");
      return;
    }
    case CARBON_KIND(FloatType inst): {
      AddIntOrFloatTypeName('f', inst.bit_width_id);
      return;
    }
    case CARBON_KIND(FunctionDecl inst): {
      const auto& function_info = sem_ir().functions().Get(inst.function_id);
      AddInstNameId(function_info.name_id, ".decl");
      auto function_scope_id = inst_namer_->GetScopeFor(inst.function_id);
      QueueBlockId(function_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(FunctionType inst): {
      AddInstNameId(sem_ir().functions().Get(inst.function_id).name_id,
                    ".type");
      return;
    }
    case CARBON_KIND(GenericClassType inst): {
      AddInstNameId(sem_ir().classes().Get(inst.class_id).name_id, ".type");
      return;
    }
    case CARBON_KIND(GenericInterfaceType inst): {
      AddInstNameId(sem_ir().interfaces().Get(inst.interface_id).name_id,
                    ".type");
      return;
    }
    case CARBON_KIND(ImplDecl inst): {
      auto impl_scope_id = inst_namer_->GetScopeFor(inst.impl_id);
      QueueBlockId(impl_scope_id, inst.decl_block_id);
      return;
    }
    case CARBON_KIND(LookupImplWitness inst): {
      const auto& interface =
          sem_ir().specific_interfaces().Get(inst.query_specific_interface_id);
      AddInstNameId(sem_ir().interfaces().Get(interface.interface_id).name_id,
                    ".lookup_impl_witness");
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
        if (!inst_namer_->insts_[const_inst_id.index].second) {
          inst_namer_->QueueBlockInsts(*queue_, ScopeId::Imports,
                                       llvm::ArrayRef(const_inst_id));
        }
      }
      return;
    }
    case CARBON_KIND(InstValue inst): {
      inst_namer_->QueueBlockInsts(*queue_, scope_id_, inst.inst_id);
      AddInstName(
          ("inst." + sem_ir().insts().Get(inst.inst_id).kind().ir_name())
              .str());
      return;
    }
    case CARBON_KIND(InterfaceDecl inst): {
      const auto& interface_info = sem_ir().interfaces().Get(inst.interface_id);
      AddInstNameId(interface_info.name_id, ".decl");
      auto interface_scope_id = inst_namer_->GetScopeFor(inst.interface_id);
      QueueBlockId(interface_scope_id, inst.decl_block_id);
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
      QueueBlockId(scope_id_, inst.pattern_block_id);
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
        AddInstNameId(sem_ir().functions().Get(fn_ty->function_id).name_id,
                      ".specific_fn");
      } else {
        AddInstName("specific_fn");
      }
      return;
    }
    case CARBON_KIND(SpecificImplFunction inst): {
      auto type_id = sem_ir().insts().Get(inst.callee_id).type_id();
      if (auto fn_ty = sem_ir().types().TryGetAs<FunctionType>(type_id)) {
        AddInstNameId(sem_ir().functions().Get(fn_ty->function_id).name_id,
                      ".specific_impl_fn");
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
      QueueBlockId(scope_id_, inst.block_id);
      AddInstName("");
      return;
    }
    case StringLiteral::Kind: {
      AddInstName("str");
      return;
    }
    case CARBON_KIND(StructValue inst): {
      if (auto fn_ty = sem_ir().types().TryGetAs<FunctionType>(inst.type_id)) {
        AddInstNameId(sem_ir().functions().Get(fn_ty->function_id).name_id);
      } else if (auto class_ty =
                     sem_ir().types().TryGetAs<ClassType>(inst.type_id)) {
        AddInstNameId(sem_ir().classes().Get(class_ty->class_id).name_id,
                      ".val");
      } else if (auto generic_class_ty =
                     sem_ir().types().TryGetAs<GenericClassType>(
                         inst.type_id)) {
        AddInstNameId(
            sem_ir().classes().Get(generic_class_ty->class_id).name_id,
            ".generic");
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
        AddInstNameId(sem_ir().classes().Get(class_ty->class_id).name_id,
                      ".elem");
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

}  // namespace Carbon::SemIR
