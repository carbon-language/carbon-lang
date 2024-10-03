// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_namer.h"

#include "common/ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

InstNamer::InstNamer(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir)
    : tokenized_buffer_(tokenized_buffer),
      parse_tree_(parse_tree),
      sem_ir_(sem_ir) {
  insts_.resize(sem_ir.insts().size(), {ScopeId::None, Namespace::Name()});
  labels_.resize(sem_ir.inst_blocks().size());
  scopes_.resize(static_cast<size_t>(GetScopeFor(NumberOfScopesTag())));
  generic_scopes_.resize(sem_ir.generics().size(), ScopeId::None);

  // Build the constants scope.
  CollectNamesInBlock(ScopeId::Constants, sem_ir.constants().array_ref());

  // Build the ImportRef scope.
  CollectNamesInBlock(ScopeId::ImportRefs,
                      sem_ir.inst_blocks().Get(SemIR::InstBlockId::ImportRefs));

  // Build the file scope.
  CollectNamesInBlock(ScopeId::File, sem_ir.top_inst_block_id());

  // Build each function scope.
  for (auto [i, fn] : llvm::enumerate(sem_ir.functions().array_ref())) {
    FunctionId fn_id(i);
    auto fn_scope = GetScopeFor(fn_id);
    // TODO: Provide a location for the function for use as a
    // disambiguator.
    auto fn_loc = Parse::NodeId::Invalid;
    GetScopeInfo(fn_scope).name = globals_.AllocateName(
        *this, fn_loc, sem_ir.names().GetIRBaseName(fn.name_id).str());
    CollectNamesInBlock(fn_scope, fn.implicit_param_refs_id);
    CollectNamesInBlock(fn_scope, fn.param_refs_id);
    if (fn.return_storage_id.is_valid()) {
      insts_[fn.return_storage_id.index] = {
          fn_scope,
          GetScopeInfo(fn_scope).insts.AllocateName(
              *this, sem_ir.insts().GetLocId(fn.return_storage_id), "return")};
    }
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
  for (auto [i, class_info] : llvm::enumerate(sem_ir.classes().array_ref())) {
    ClassId class_id(i);
    auto class_scope = GetScopeFor(class_id);
    // TODO: Provide a location for the class for use as a disambiguator.
    auto class_loc = Parse::NodeId::Invalid;
    GetScopeInfo(class_scope).name = globals_.AllocateName(
        *this, class_loc,
        sem_ir.names().GetIRBaseName(class_info.name_id).str());
    AddBlockLabel(class_scope, class_info.body_block_id, "class", class_loc);
    CollectNamesInBlock(class_scope, class_info.body_block_id);
    CollectNamesInGeneric(class_scope, class_info.generic_id);
  }

  // Build each interface scope.
  for (auto [i, interface_info] :
       llvm::enumerate(sem_ir.interfaces().array_ref())) {
    InterfaceId interface_id(i);
    auto interface_scope = GetScopeFor(interface_id);
    // TODO: Provide a location for the interface for use as a disambiguator.
    auto interface_loc = Parse::NodeId::Invalid;
    GetScopeInfo(interface_scope).name = globals_.AllocateName(
        *this, interface_loc,
        sem_ir.names().GetIRBaseName(interface_info.name_id).str());
    AddBlockLabel(interface_scope, interface_info.body_block_id, "interface",
                  interface_loc);
    CollectNamesInBlock(interface_scope, interface_info.body_block_id);
    CollectNamesInGeneric(interface_scope, interface_info.generic_id);
  }

  // Build each impl scope.
  for (auto [i, impl_info] : llvm::enumerate(sem_ir.impls().array_ref())) {
    ImplId impl_id(i);
    auto impl_scope = GetScopeFor(impl_id);
    // TODO: Provide a location for the impl for use as a disambiguator.
    auto impl_loc = Parse::NodeId::Invalid;
    // TODO: Invent a name based on the self and constraint types.
    GetScopeInfo(impl_scope).name =
        globals_.AllocateName(*this, impl_loc, "impl");
    AddBlockLabel(impl_scope, impl_info.body_block_id, "impl", impl_loc);
    CollectNamesInBlock(impl_scope, impl_info.body_block_id);
    CollectNamesInGeneric(impl_scope, impl_info.generic_id);
  }
}

auto InstNamer::GetScopeName(ScopeId scope) const -> std::string {
  switch (scope) {
    case ScopeId::None:
      return "<invalid scope>";

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
  if (!inst_id.is_valid()) {
    return "";
  }
  const auto& inst_name = insts_[inst_id.index].second;
  return inst_name ? inst_name.str() : "";
}

auto InstNamer::GetNameFor(ScopeId scope_id, InstId inst_id) const
    -> std::string {
  if (!inst_id.is_valid()) {
    return "invalid";
  }

  // Check for a builtin.
  if (inst_id.is_builtin()) {
    return inst_id.builtin_inst_kind().label().str();
  }

  if (inst_id == InstId::PackageNamespace) {
    return "package";
  }

  const auto& [inst_scope, inst_name] = insts_[inst_id.index];
  if (!inst_name) {
    // This should not happen in valid IR.
    std::string str;
    llvm::raw_string_ostream str_stream(str);
    str_stream << "<unexpected>." << inst_id;
    auto loc_id = sem_ir_.insts().GetLocId(inst_id);
    // TODO: Consider handling inst_id cases.
    if (loc_id.is_node_id()) {
      auto token = parse_tree_.node_token(loc_id.node_id());
      str_stream << ".loc" << tokenized_buffer_.GetLineNumber(token) << "_"
                 << tokenized_buffer_.GetColumnNumber(token);
    }
    return str;
  }
  if (inst_scope == scope_id) {
    return ("%" + inst_name.str()).str();
  }
  return (GetScopeName(inst_scope) + ".%" + inst_name.str()).str();
}

auto InstNamer::GetUnscopedLabelFor(InstBlockId block_id) const
    -> llvm::StringRef {
  if (!block_id.is_valid()) {
    return "";
  }
  const auto& label_name = labels_[block_id.index].second;
  return label_name ? label_name.str() : "";
}

// Returns the IR name to use for a label, when referenced from a given scope.
auto InstNamer::GetLabelFor(ScopeId scope_id, InstBlockId block_id) const
    -> std::string {
  if (!block_id.is_valid()) {
    return "!invalid";
  }

  const auto& [label_scope, label_name] = labels_[block_id.index];
  if (!label_name) {
    // This should not happen in valid IR.
    std::string str;
    llvm::raw_string_ostream(str)
        << "<unexpected instblockref " << block_id << ">";
    return str;
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

auto InstNamer::Namespace::AllocateName(const InstNamer& inst_namer,
                                        SemIR::LocId loc_id, std::string name)
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
  // TODO: Consider handling inst_id cases.
  if (loc_id.is_node_id()) {
    auto token = inst_namer.parse_tree_.node_token(loc_id.node_id());
    llvm::raw_string_ostream(name)
        << ".loc" << inst_namer.tokenized_buffer_.GetLineNumber(token);
    add_name();

    llvm::raw_string_ostream(name)
        << "_" << inst_namer.tokenized_buffer_.GetColumnNumber(token);
    add_name();
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

auto InstNamer::AddBlockLabel(ScopeId scope_id, InstBlockId block_id,
                              std::string name, SemIR::LocId loc_id) -> void {
  if (!block_id.is_valid() || labels_[block_id.index].second) {
    return;
  }

  if (!loc_id.is_valid()) {
    if (const auto& block = sem_ir_.inst_blocks().Get(block_id);
        !block.empty()) {
      loc_id = sem_ir_.insts().GetLocId(block.front());
    }
  }

  labels_[block_id.index] = {
      scope_id, GetScopeInfo(scope_id).labels.AllocateName(*this, loc_id,
                                                           std::move(name))};
}

// Finds and adds a suitable block label for the given SemIR instruction that
// represents some kind of branch.
auto InstNamer::AddBlockLabel(ScopeId scope_id, SemIR::LocId loc_id,
                              AnyBranch branch) -> void {
  llvm::StringRef name;
  switch (parse_tree_.node_kind(loc_id.node_id())) {
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
  if (block_id.is_valid()) {
    CollectNamesInBlock(scope_id, sem_ir_.inst_blocks().Get(block_id));
  }
}

auto InstNamer::CollectNamesInBlock(ScopeId scope_id,
                                    llvm::ArrayRef<InstId> block) -> void {
  Scope& scope = GetScopeInfo(scope_id);

  // Use bound names where available. Otherwise, assign a backup name.
  for (auto inst_id : block) {
    if (!inst_id.is_valid()) {
      continue;
    }

    auto untyped_inst = sem_ir_.insts().Get(inst_id);
    auto add_inst_name = [&](std::string name) {
      ScopeId old_scope_id = insts_[inst_id.index].first;
      if (old_scope_id == ScopeId::None) {
        insts_[inst_id.index] = {
            scope_id, scope.insts.AllocateName(
                          *this, sem_ir_.insts().GetLocId(inst_id), name)};
      } else {
        CARBON_CHECK(old_scope_id == scope_id,
                     "Attempting to name inst in multiple scopes");
      }
    };
    auto add_inst_name_id = [&](NameId name_id, llvm::StringRef suffix = "") {
      add_inst_name(
          (sem_ir_.names().GetIRBaseName(name_id).str() + suffix).str());
    };

    if (auto branch = untyped_inst.TryAs<AnyBranch>()) {
      AddBlockLabel(scope_id, sem_ir_.insts().GetLocId(inst_id), *branch);
    }

    CARBON_KIND_SWITCH(untyped_inst) {
      case CARBON_KIND(AddrPattern inst): {
        // TODO: We need to assign names to parameters that appear in
        // function declarations, which may be nested within a pattern. For
        // now, just look through `addr`, but we should find a better way to
        // visit parameters.
        CollectNamesInBlock(scope_id, inst.inner_id);
        break;
      }
      case CARBON_KIND(AssociatedConstantDecl inst): {
        add_inst_name_id(inst.name_id);
        continue;
      }
      case BindAlias::Kind:
      case BindName::Kind:
      case BindSymbolicName::Kind:
      case ExportDecl::Kind: {
        auto inst = untyped_inst.As<AnyBindNameOrExportDecl>();
        add_inst_name_id(
            sem_ir_.entity_names().Get(inst.entity_name_id).name_id);
        continue;
      }
      case BindingPattern::Kind:
      case SymbolicBindingPattern::Kind: {
        auto inst = untyped_inst.As<AnyBindingPattern>();
        add_inst_name_id(
            sem_ir_.entity_names().Get(inst.entity_name_id).name_id, ".patt");
        continue;
      }
      case CARBON_KIND(Call inst): {
        auto callee_function =
            SemIR::GetCalleeFunction(sem_ir_, inst.callee_id);
        if (!callee_function.function_id.is_valid()) {
          break;
        }
        const auto& function =
            sem_ir_.functions().Get(callee_function.function_id);
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
        const auto& class_info = sem_ir_.classes().Get(inst.class_id);
        add_inst_name_id(class_info.name_id, ".decl");
        auto class_scope_id = GetScopeFor(inst.class_id);
        CollectNamesInBlock(class_scope_id, class_info.pattern_block_id);
        CollectNamesInBlock(class_scope_id, inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(ClassType inst): {
        add_inst_name_id(sem_ir_.classes().Get(inst.class_id).name_id);
        continue;
      }
      case CARBON_KIND(FunctionDecl inst): {
        const auto& function_info = sem_ir_.functions().Get(inst.function_id);
        add_inst_name_id(function_info.name_id, ".decl");
        auto function_scope_id = GetScopeFor(inst.function_id);
        CollectNamesInBlock(function_scope_id, function_info.pattern_block_id);
        CollectNamesInBlock(function_scope_id, inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(FunctionType inst): {
        add_inst_name_id(sem_ir_.functions().Get(inst.function_id).name_id,
                         ".type");
        continue;
      }
      case CARBON_KIND(GenericClassType inst): {
        add_inst_name_id(sem_ir_.classes().Get(inst.class_id).name_id, ".type");
        continue;
      }
      case CARBON_KIND(GenericInterfaceType inst): {
        add_inst_name_id(sem_ir_.interfaces().Get(inst.interface_id).name_id,
                         ".type");
        continue;
      }
      case CARBON_KIND(ImplDecl inst): {
        auto impl_scope_id = GetScopeFor(inst.impl_id);
        CollectNamesInBlock(impl_scope_id,
                            sem_ir_.impls().Get(inst.impl_id).pattern_block_id);
        CollectNamesInBlock(impl_scope_id, inst.decl_block_id);
        break;
      }
      case CARBON_KIND(ImportDecl inst): {
        if (inst.package_id.is_valid()) {
          add_inst_name_id(inst.package_id, ".import");
        } else {
          add_inst_name("default.import");
        }
        break;
      }
      case ImportRefUnloaded::Kind:
      case ImportRefLoaded::Kind: {
        add_inst_name("import_ref");
        // When building import refs, we frequently add instructions without
        // a block. Constants that refer to them need to be separately
        // named.
        auto const_id = sem_ir_.constant_values().Get(inst_id);
        if (const_id.is_valid() && const_id.is_template()) {
          auto const_inst_id = sem_ir_.constant_values().GetInstId(const_id);
          if (!insts_[const_inst_id.index].second) {
            CollectNamesInBlock(ScopeId::ImportRefs, const_inst_id);
          }
        }
        continue;
      }
      case CARBON_KIND(InterfaceDecl inst): {
        const auto& interface_info =
            sem_ir_.interfaces().Get(inst.interface_id);
        add_inst_name_id(interface_info.name_id, ".decl");
        auto interface_scope_id = GetScopeFor(inst.interface_id);
        CollectNamesInBlock(interface_scope_id,
                            interface_info.pattern_block_id);
        CollectNamesInBlock(interface_scope_id, inst.decl_block_id);
        continue;
      }
      case CARBON_KIND(InterfaceType inst): {
        const auto& interface_info =
            sem_ir_.interfaces().Get(inst.interface_id);
        add_inst_name_id(interface_info.name_id, ".type");
        continue;
      }
      case CARBON_KIND(NameRef inst): {
        add_inst_name_id(inst.name_id, ".ref");
        continue;
      }
      // The namespace is specified here due to the name conflict.
      case CARBON_KIND(SemIR::Namespace inst): {
        add_inst_name_id(sem_ir_.name_scopes().Get(inst.name_scope_id).name_id);
        continue;
      }
      case CARBON_KIND(Param inst): {
        add_inst_name_id(inst.name_id, ".param");
        continue;
      }
      case CARBON_KIND(SpliceBlock inst): {
        CollectNamesInBlock(scope_id, inst.block_id);
        break;
      }
      case CARBON_KIND(StructValue inst): {
        if (auto fn_ty = sem_ir_.types().TryGetAs<FunctionType>(inst.type_id)) {
          add_inst_name_id(sem_ir_.functions().Get(fn_ty->function_id).name_id);
        } else if (auto generic_class_ty =
                       sem_ir_.types().TryGetAs<GenericClassType>(
                           inst.type_id)) {
          add_inst_name_id(
              sem_ir_.classes().Get(generic_class_ty->class_id).name_id);
        } else if (auto generic_interface_ty =
                       sem_ir_.types().TryGetAs<GenericInterfaceType>(
                           inst.type_id)) {
          add_inst_name_id(sem_ir_.interfaces()
                               .Get(generic_interface_ty->interface_id)
                               .name_id);
        } else {
          add_inst_name("struct");
        }
        continue;
      }
      case CARBON_KIND(TupleValue inst): {
        if (sem_ir_.types().Is<ArrayType>(inst.type_id)) {
          add_inst_name("array");
        } else {
          add_inst_name("tuple");
        }
        continue;
      }
      case CARBON_KIND(VarStorage inst): {
        add_inst_name_id(inst.name_id, ".var");
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
  if (!generic_id.is_valid()) {
    return;
  }
  generic_scopes_[generic_id.index] = scope_id;
  const auto& generic = sem_ir_.generics().Get(generic_id);
  CollectNamesInBlock(scope_id, generic.decl_block_id);
  CollectNamesInBlock(scope_id, generic.definition_block_id);
}

}  // namespace Carbon::SemIR
