// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_namer.h"

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

InstNamer::InstNamer(const File* sem_ir) : sem_ir_(sem_ir) {
  insts_.resize(sem_ir->insts().size(), {ScopeId::None, Namespace::Name()});
  labels_.resize(sem_ir->inst_blocks().size());
  scopes_.resize(static_cast<size_t>(GetScopeFor(NumberOfScopesTag())));
  generic_scopes_.resize(sem_ir->generics().size(), ScopeId::None);

  // Build the constants scope.
  CollectNamesInBlock(ScopeId::Constants, sem_ir->constants().array_ref());

  // Build the ImportRef scope.
  CollectNamesInBlock(ScopeId::ImportRefs, sem_ir->inst_blocks().Get(
                                               SemIR::InstBlockId::ImportRefs));

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
    CollectNamesInBlock(fn_scope, fn.implicit_param_patterns_id);
    CollectNamesInBlock(fn_scope, fn.param_patterns_id);
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
    AddBlockLabel(interface_scope, interface_info.body_block_id, "interface",
                  interface_loc);
    CollectNamesInBlock(interface_scope, interface_info.body_block_id);
    CollectNamesInGeneric(interface_scope, interface_info.generic_id);
  }

  // Build each associated constant scope.
  for (auto [assoc_const_id, assoc_const_info] :
       sem_ir->associated_constants().enumerate()) {
    auto assoc_const_scope = GetScopeFor(assoc_const_id);
    auto assoc_const_loc = sem_ir->insts().GetLocId(assoc_const_info.decl_id);
    GetScopeInfo(assoc_const_scope).name = globals_.AllocateName(
        *this, assoc_const_loc,
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
    AddBlockLabel(impl_scope, impl_info.body_block_id, "impl",
                  impl_fingerprint);
    CollectNamesInBlock(impl_scope, impl_info.body_block_id);
    CollectNamesInGeneric(impl_scope, impl_info.generic_id);
  }
}

auto InstNamer::GetScopeName(ScopeId scope) const -> std::string {
  switch (scope) {
    case ScopeId::None:
      return "<no scope>";

    // These are treated as SemIR keywords.
    case ScopeId::File:
      return "file";
    case ScopeId::ImportRefs:
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
  if (SemIR::IsSingletonInstId(inst_id)) {
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
    auto loc_id = sem_ir_->insts().GetLocId(inst_id);
    // TODO: Consider handling inst_id cases.
    if (loc_id.is_node_id()) {
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
    std::variant<SemIR::LocId, uint64_t> loc_id_or_fingerprint,
    std::string name) -> Name {
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
  // TODO: Consider handling inst_id cases.
  if (auto* loc_id = std::get_if<LocId>(&loc_id_or_fingerprint)) {
    if (loc_id->is_node_id()) {
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
    std::variant<SemIR::LocId, uint64_t> loc_id_or_fingerprint) -> void {
  if (!block_id.has_value() || labels_[block_id.index].second) {
    return;
  }

  if (auto* loc_id = std::get_if<LocId>(&loc_id_or_fingerprint);
      loc_id && !loc_id->has_value()) {
    if (const auto& block = sem_ir_->inst_blocks().Get(block_id);
        !block.empty()) {
      loc_id_or_fingerprint = sem_ir_->insts().GetLocId(block.front());
    }
  }

  labels_[block_id.index] = {
      scope_id, GetScopeInfo(scope_id).labels.AllocateName(
                    *this, loc_id_or_fingerprint, std::move(name))};
}

// Finds and adds a suitable block label for the given SemIR instruction that
// represents some kind of branch.
auto InstNamer::AddBlockLabel(ScopeId scope_id, SemIR::LocId loc_id,
                              AnyBranch branch) -> void {
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
  llvm::SmallVector<std::pair<ScopeId, InstId>> insts;

  // Adds a scope and instructions to walk. Avoids recursion while allowing
  // the loop to below add more instructions during iteration. The new
  // instructions are queued such that they will be the next to be walked.
  // Internally that means they are reversed and added to the end of the vector,
  // since we pop from the back of the vector.
  auto queue_block_insts = [&](ScopeId scope_id,
                               llvm::ArrayRef<InstId> inst_ids) {
    for (auto inst_id : llvm::reverse(inst_ids)) {
      if (inst_id.has_value() && !SemIR::IsSingletonInstId(inst_id)) {
        insts.push_back(std::make_pair(scope_id, inst_id));
      }
    }
  };
  auto queue_block_id = [&](ScopeId scope_id, InstBlockId block_id) {
    if (block_id.has_value()) {
      queue_block_insts(scope_id, sem_ir_->inst_blocks().Get(block_id));
    }
  };

  queue_block_insts(top_scope_id, block);

  // Use bound names where available. Otherwise, assign a backup name.
  while (!insts.empty()) {
    auto [scope_id, inst_id] = insts.pop_back_val();

    Scope& scope = GetScopeInfo(scope_id);

    auto untyped_inst = sem_ir_->insts().Get(inst_id);
    auto add_inst_name = [&](std::string name) {
      ScopeId old_scope_id = insts_[inst_id.index].first;
      if (old_scope_id == ScopeId::None) {
        std::variant<SemIR::LocId, uint64_t> loc_id_or_fingerprint =
            SemIR::LocId::None;
        if (scope_id == ScopeId::Constants || scope_id == ScopeId::ImportRefs) {
          loc_id_or_fingerprint = fingerprinter_.GetOrCompute(sem_ir_, inst_id);
        } else {
          loc_id_or_fingerprint = sem_ir_->insts().GetLocId(inst_id);
        }
        insts_[inst_id.index] = {
            scope_id,
            scope.insts.AllocateName(*this, loc_id_or_fingerprint, name)};
      } else {
        CARBON_CHECK(old_scope_id == scope_id,
                     "Attempting to name inst in multiple scopes");
      }
    };
    auto add_inst_name_id = [&](NameId name_id, llvm::StringRef suffix = "") {
      add_inst_name(
          (sem_ir_->names().GetIRBaseName(name_id).str() + suffix).str());
    };
    auto add_int_or_float_type_name = [&](char type_literal_prefix,
                                          SemIR::InstId bit_width_id,
                                          llvm::StringRef suffix = "") {
      RawStringOstream out;
      out << type_literal_prefix;
      if (auto bit_width = sem_ir_->insts().TryGetAs<IntValue>(bit_width_id)) {
        out << sem_ir_->ints().Get(bit_width->int_id);
      } else {
        out << "N";
      }
      out << suffix;
      add_inst_name(out.TakeStr());
    };
    auto facet_access_name_id = [&](InstId facet_value_inst_id) -> NameId {
      if (auto name = sem_ir_->insts().TryGetAs<NameRef>(facet_value_inst_id)) {
        return name->name_id;
      } else if (auto symbolic = sem_ir_->insts().TryGetAs<BindSymbolicName>(
                     facet_value_inst_id)) {
        return sem_ir_->entity_names().Get(symbolic->entity_name_id).name_id;
      }
      return NameId::None;
    };

    if (auto branch = untyped_inst.TryAs<AnyBranch>()) {
      AddBlockLabel(scope_id, sem_ir_->insts().GetLocId(inst_id), *branch);
    }

    CARBON_KIND_SWITCH(untyped_inst) {
      case AddrOf::Kind: {
        add_inst_name("addr");
        continue;
      }
      case ArrayType::Kind: {
        // TODO: Can we figure out the name of the type this is an array of?
        add_inst_name("array_type");
        continue;
      }
      case CARBON_KIND(AssociatedConstantDecl inst): {
        add_inst_name_id(
            sem_ir_->associated_constants().Get(inst.assoc_const_id).name_id);
        queue_block_id(GetScopeFor(inst.assoc_const_id), inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(AssociatedEntity inst): {
        RawStringOstream out;
        out << "assoc" << inst.index.index;
        add_inst_name(out.TakeStr());
        continue;
      }
      case CARBON_KIND(AssociatedEntityType inst): {
        auto facet_type =
            sem_ir_->types().TryGetAs<FacetType>(inst.interface_type_id);
        if (!facet_type) {
          // Should never happen, but we don't want the instruction namer to
          // crash on bad IR.
          add_inst_name("<invalid interface>");
          continue;
        }
        const auto& facet_type_info =
            sem_ir_->facet_types().Get(facet_type->facet_type_id);
        auto interface = facet_type_info.TryAsSingleInterface();
        if (!interface) {
          // Should never happen, but we don't want the instruction namer to
          // crash on bad IR.
          add_inst_name("<invalid interface>");
          continue;
        }
        const auto& interface_info =
            sem_ir_->interfaces().Get(interface->interface_id);
        add_inst_name_id(interface_info.name_id, ".assoc_type");
        continue;
      }
      case BindAlias::Kind:
      case BindName::Kind:
      case BindSymbolicName::Kind:
      case ExportDecl::Kind: {
        auto inst = untyped_inst.As<AnyBindNameOrExportDecl>();
        add_inst_name_id(
            sem_ir_->entity_names().Get(inst.entity_name_id).name_id);
        continue;
      }
      case BindingPattern::Kind:
      case SymbolicBindingPattern::Kind: {
        auto inst = untyped_inst.As<AnyBindingPattern>();
        auto name_id = NameId::Underscore;
        if (inst.entity_name_id.has_value()) {
          name_id = sem_ir_->entity_names().Get(inst.entity_name_id).name_id;
        }
        add_inst_name_id(name_id, ".patt");
        continue;
      }
      case CARBON_KIND(BoolLiteral inst): {
        if (inst.value.ToBool()) {
          add_inst_name("true");
        } else {
          add_inst_name("false");
        }
        continue;
      }
      case CARBON_KIND(BoundMethod inst): {
        auto type_id = sem_ir_->insts().Get(inst.function_decl_id).type_id();
        if (auto fn_ty = sem_ir_->types().TryGetAs<FunctionType>(type_id)) {
          add_inst_name_id(sem_ir_->functions().Get(fn_ty->function_id).name_id,
                           ".bound");
        } else {
          add_inst_name("bound_method");
        }
        continue;
      }
      case CARBON_KIND(Call inst): {
        auto callee_function =
            SemIR::GetCalleeFunction(*sem_ir_, inst.callee_id);
        if (!callee_function.function_id.has_value()) {
          break;
        }
        const auto& function =
            sem_ir_->functions().Get(callee_function.function_id);
        // Name the call's result based on the callee.
        if (function.builtin_function_kind !=
            SemIR::BuiltinFunctionKind::None) {
          // For a builtin, use the builtin name. Otherwise, we'd typically pick
          // the name `Op` below, which is probably not very useful.
          add_inst_name(function.builtin_function_kind.name().str());
          continue;
        }

        add_inst_name_id(function.name_id, ".call");
        continue;
      }
      case CARBON_KIND(ClassDecl inst): {
        const auto& class_info = sem_ir_->classes().Get(inst.class_id);
        add_inst_name_id(class_info.name_id, ".decl");
        auto class_scope_id = GetScopeFor(inst.class_id);
        queue_block_id(class_scope_id, class_info.pattern_block_id);
        queue_block_id(class_scope_id, inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(ClassType inst): {
        if (auto literal_info = NumericTypeLiteralInfo::ForType(*sem_ir_, inst);
            literal_info.is_valid()) {
          add_inst_name(literal_info.GetLiteralAsString(*sem_ir_));
          break;
        }
        add_inst_name_id(sem_ir_->classes().Get(inst.class_id).name_id);
        continue;
      }
      case CompleteTypeWitness::Kind: {
        // TODO: Can we figure out the name of the type this is a witness for?
        add_inst_name("complete_type");
        continue;
      }
      case ConstType::Kind: {
        // TODO: Can we figure out the name of the type argument?
        add_inst_name("const");
        continue;
      }
      case CARBON_KIND(FacetAccessType inst): {
        auto name_id = facet_access_name_id(inst.facet_value_inst_id);
        if (name_id.has_value()) {
          add_inst_name_id(name_id, ".as_type");
        } else {
          add_inst_name("as_type");
        }
        continue;
      }
      case CARBON_KIND(FacetAccessWitness inst): {
        auto name_id = facet_access_name_id(inst.facet_value_inst_id);
        RawStringOstream out;
        if (name_id.has_value()) {
          out << ".as_wit.iface" << inst.index.index;
          add_inst_name_id(name_id, out.TakeStr());
        } else {
          out << "as_wit.iface" << inst.index.index;
          add_inst_name(out.TakeStr());
        }
        continue;
      }
      case CARBON_KIND(FacetType inst): {
        const auto& facet_type_info =
            sem_ir_->facet_types().Get(inst.facet_type_id);
        bool has_where = facet_type_info.other_requirements ||
                         !facet_type_info.rewrite_constraints.empty();
        if (auto interface = facet_type_info.TryAsSingleInterface()) {
          const auto& interface_info =
              sem_ir_->interfaces().Get(interface->interface_id);
          add_inst_name_id(interface_info.name_id,
                           has_where ? "_where.type" : ".type");
        } else if (facet_type_info.impls_constraints.empty()) {
          add_inst_name(has_where ? "type_where" : "type");
        } else {
          add_inst_name("facet_type");
        }
        continue;
      }
      case CARBON_KIND(FacetValue inst): {
        if (auto facet_type =
                sem_ir_->types().TryGetAs<FacetType>(inst.type_id)) {
          const auto& facet_type_info =
              sem_ir_->facet_types().Get(facet_type->facet_type_id);
          if (auto interface = facet_type_info.TryAsSingleInterface()) {
            const auto& interface_info =
                sem_ir_->interfaces().Get(interface->interface_id);
            add_inst_name_id(interface_info.name_id, ".facet");
            continue;
          }
        }
        add_inst_name("facet_value");
        continue;
      }
      case FloatLiteral::Kind: {
        add_inst_name("float");
        continue;
      }
      case CARBON_KIND(FloatType inst): {
        add_int_or_float_type_name('f', inst.bit_width_id);
        continue;
      }
      case CARBON_KIND(FunctionDecl inst): {
        const auto& function_info = sem_ir_->functions().Get(inst.function_id);
        add_inst_name_id(function_info.name_id, ".decl");
        auto function_scope_id = GetScopeFor(inst.function_id);
        queue_block_id(function_scope_id, function_info.pattern_block_id);
        queue_block_id(function_scope_id, inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(FunctionType inst): {
        add_inst_name_id(sem_ir_->functions().Get(inst.function_id).name_id,
                         ".type");
        continue;
      }
      case CARBON_KIND(GenericClassType inst): {
        add_inst_name_id(sem_ir_->classes().Get(inst.class_id).name_id,
                         ".type");
        continue;
      }
      case CARBON_KIND(GenericInterfaceType inst): {
        add_inst_name_id(sem_ir_->interfaces().Get(inst.interface_id).name_id,
                         ".type");
        continue;
      }
      case CARBON_KIND(ImplDecl inst): {
        auto impl_scope_id = GetScopeFor(inst.impl_id);
        queue_block_id(impl_scope_id,
                       sem_ir_->impls().Get(inst.impl_id).pattern_block_id);
        queue_block_id(impl_scope_id, inst.decl_block_id);
        break;
      }
      case ImplWitness::Kind: {
        // TODO: Include name of interface (is this available from the
        // specific?).
        add_inst_name("impl_witness");
        continue;
      }
      case CARBON_KIND(ImplWitnessAccess inst): {
        // TODO: Include information about the impl?
        RawStringOstream out;
        out << "impl.elem" << inst.index.index;
        add_inst_name(out.TakeStr());
        continue;
      }
      case ImportCppDecl::Kind: {
        add_inst_name("Cpp.import_cpp");
        continue;
      }
      case CARBON_KIND(ImportDecl inst): {
        if (inst.package_id.has_value()) {
          add_inst_name_id(inst.package_id, ".import");
        } else {
          add_inst_name("default.import");
        }
        continue;
      }
      case ImportRefUnloaded::Kind:
      case ImportRefLoaded::Kind: {
        // Build the base import name: <package>.<entity-name>
        RawStringOstream out;

        auto inst = untyped_inst.As<AnyImportRef>();
        auto import_ir_inst =
            sem_ir_->import_ir_insts().Get(inst.import_ir_inst_id);
        const auto& import_ir =
            *sem_ir_->import_irs().Get(import_ir_inst.ir_id).sem_ir;
        auto package_id = import_ir.package_id();
        if (auto ident_id = package_id.AsIdentifierId(); ident_id.has_value()) {
          out << import_ir.identifiers().Get(ident_id);
        } else {
          out << package_id.AsSpecialName();
        }
        out << ".";

        // Add entity name if available.
        if (inst.entity_name_id.has_value()) {
          auto name_id =
              sem_ir_->entity_names().Get(inst.entity_name_id).name_id;
          out << sem_ir_->names().GetIRBaseName(name_id);
        } else {
          out << "import_ref";
        }

        add_inst_name(out.TakeStr());

        // When building import refs, we frequently add instructions without
        // a block. Constants that refer to them need to be separately
        // named.
        auto const_id = sem_ir_->constant_values().Get(inst_id);
        if (const_id.has_value() && const_id.is_concrete()) {
          auto const_inst_id = sem_ir_->constant_values().GetInstId(const_id);
          if (!insts_[const_inst_id.index].second) {
            queue_block_insts(ScopeId::ImportRefs,
                              llvm::ArrayRef(const_inst_id));
          }
        }
        continue;
      }
      case CARBON_KIND(InterfaceDecl inst): {
        const auto& interface_info =
            sem_ir_->interfaces().Get(inst.interface_id);
        add_inst_name_id(interface_info.name_id, ".decl");
        auto interface_scope_id = GetScopeFor(inst.interface_id);
        queue_block_id(interface_scope_id, interface_info.pattern_block_id);
        queue_block_id(interface_scope_id, inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(IntType inst): {
        add_int_or_float_type_name(inst.int_kind == IntKind::Signed ? 'i' : 'u',
                                   inst.bit_width_id, ".builtin");
        continue;
      }
      case CARBON_KIND(IntValue inst): {
        RawStringOstream out;
        out << "int_" << sem_ir_->ints().Get(inst.int_id);
        add_inst_name(out.TakeStr());
        continue;
      }
      case CARBON_KIND(NameBindingDecl inst): {
        queue_block_id(scope_id, inst.pattern_block_id);
        continue;
      }
      case CARBON_KIND(NameRef inst): {
        add_inst_name_id(inst.name_id, ".ref");
        continue;
      }
      // The namespace is specified here due to the name conflict.
      case CARBON_KIND(SemIR::Namespace inst): {
        add_inst_name_id(
            sem_ir_->name_scopes().Get(inst.name_scope_id).name_id());
        continue;
      }
      case OutParam::Kind:
      case RefParam::Kind:
      case ValueParam::Kind: {
        add_inst_name_id(untyped_inst.As<AnyParam>().pretty_name_id, ".param");
        continue;
      }
      case OutParamPattern::Kind:
      case RefParamPattern::Kind:
      case ValueParamPattern::Kind: {
        add_inst_name_id(SemIR::GetPrettyNameFromPatternId(*sem_ir_, inst_id),
                         ".param_patt");
        continue;
      }
      case PointerType::Kind: {
        add_inst_name("ptr");
        continue;
      }
      case RequireCompleteType::Kind: {
        add_inst_name("require_complete");
        continue;
      }
      case ReturnSlotPattern::Kind: {
        add_inst_name_id(NameId::ReturnSlot, ".patt");
        continue;
      }
      case CARBON_KIND(SpecificFunction inst): {
        InstId callee_id = inst.callee_id;
        if (auto method = sem_ir_->insts().TryGetAs<BoundMethod>(callee_id)) {
          callee_id = method->function_decl_id;
        }
        auto type_id = sem_ir_->insts().Get(callee_id).type_id();
        if (auto fn_ty = sem_ir_->types().TryGetAs<FunctionType>(type_id)) {
          add_inst_name_id(sem_ir_->functions().Get(fn_ty->function_id).name_id,
                           ".specific_fn");
        } else {
          add_inst_name("specific_fn");
        }
        continue;
      }
      case ReturnSlot::Kind: {
        add_inst_name_id(NameId::ReturnSlot);
        break;
      }
      case CARBON_KIND(SpliceBlock inst): {
        queue_block_id(scope_id, inst.block_id);
        break;
      }
      case StringLiteral::Kind: {
        add_inst_name("str");
        continue;
      }
      case CARBON_KIND(StructValue inst): {
        if (auto fn_ty =
                sem_ir_->types().TryGetAs<FunctionType>(inst.type_id)) {
          add_inst_name_id(
              sem_ir_->functions().Get(fn_ty->function_id).name_id);
        } else if (auto class_ty =
                       sem_ir_->types().TryGetAs<ClassType>(inst.type_id)) {
          add_inst_name_id(sem_ir_->classes().Get(class_ty->class_id).name_id,
                           ".val");
        } else if (auto generic_class_ty =
                       sem_ir_->types().TryGetAs<GenericClassType>(
                           inst.type_id)) {
          add_inst_name_id(
              sem_ir_->classes().Get(generic_class_ty->class_id).name_id,
              ".generic");
        } else if (auto generic_interface_ty =
                       sem_ir_->types().TryGetAs<GenericInterfaceType>(
                           inst.type_id)) {
          add_inst_name_id(sem_ir_->interfaces()
                               .Get(generic_interface_ty->interface_id)
                               .name_id,
                           ".generic");
        } else {
          if (sem_ir_->inst_blocks().Get(inst.elements_id).empty()) {
            add_inst_name("empty_struct");
          } else {
            add_inst_name("struct");
          }
        }
        continue;
      }
      case CARBON_KIND(StructType inst): {
        const auto& fields = sem_ir_->struct_type_fields().Get(inst.fields_id);
        if (fields.empty()) {
          add_inst_name("empty_struct_type");
          continue;
        }
        std::string name = "struct_type";
        for (auto field : fields) {
          name += ".";
          name += sem_ir_->names().GetIRBaseName(field.name_id).str();
        }
        add_inst_name(std::move(name));
        continue;
      }
      case CARBON_KIND(TupleAccess inst): {
        RawStringOstream out;
        out << "tuple.elem" << inst.index.index;
        add_inst_name(out.TakeStr());
        continue;
      }
      case CARBON_KIND(TupleType inst): {
        if (inst.elements_id == TypeBlockId::Empty) {
          add_inst_name("empty_tuple.type");
        } else {
          add_inst_name("tuple.type");
        }
        continue;
      }
      case CARBON_KIND(TupleValue inst): {
        if (sem_ir_->types().Is<ArrayType>(inst.type_id)) {
          add_inst_name("array");
        } else if (inst.elements_id == InstBlockId::Empty) {
          add_inst_name("empty_tuple");
        } else {
          add_inst_name("tuple");
        }
        continue;
      }
      case CARBON_KIND(UnboundElementType inst): {
        if (auto class_ty =
                sem_ir_->types().TryGetAs<ClassType>(inst.class_type_id)) {
          add_inst_name_id(sem_ir_->classes().Get(class_ty->class_id).name_id,
                           ".elem");
        } else {
          add_inst_name("elem_type");
        }
        continue;
      }
      case CARBON_KIND(VarStorage inst): {
        add_inst_name_id(inst.pretty_name_id, ".var");
        continue;
      }
      default: {
        break;
      }
    }

    // Sequentially number all remaining values.
    if (untyped_inst.kind().value_kind() != InstValueKind::None) {
      add_inst_name("");
    }
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

}  // namespace Carbon::SemIR
