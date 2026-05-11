// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter.h"

#include <string>
#include <utility>

#include "common/ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/formatter_chunks.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/vtable.h"

// TODO: Consider addressing recursion here, although it's not critical because
// the formatter isn't required to work on arbitrary code. Still, it may help
// in the future to debug complex code.
// NOLINTBEGIN(misc-no-recursion)

namespace Carbon::SemIR {

using TentativeScopeArray =
    std::array<std::pair<InstNamer::ScopeId, llvm::ArrayRef<InstId>>,
               static_cast<size_t>(InstNamer::ScopeId::FirstEntityScope) - 1>;

// Returns blocks for the tentative scopes.
static auto GetTentativeScopes(const SemIR::File& sem_ir)
    -> TentativeScopeArray {
  return TentativeScopeArray({
      {InstNamer::ScopeId::Constants, sem_ir.constants().array_ref()},
      {InstNamer::ScopeId::Imports,
       sem_ir.inst_blocks().Get(InstBlockId::Imports)},
      {InstNamer::ScopeId::Generated,
       sem_ir.inst_blocks().Get(InstBlockId::Generated)},
  });
}

Formatter::Formatter(
    const File* sem_ir, int total_ir_count,
    Parse::GetTreeAndSubtreesFn get_tree_and_subtrees,
    const FixedSizeValueStore<CheckIRId, bool>* include_ir_in_dumps,
    bool use_dump_sem_ir_ranges)
    : sem_ir_(sem_ir),
      inst_namer_(sem_ir_, total_ir_count),
      get_tree_and_subtrees_(get_tree_and_subtrees),
      include_ir_in_dumps_(include_ir_in_dumps),
      use_dump_sem_ir_ranges_(use_dump_sem_ir_ranges),
      tentative_inst_chunks_(sem_ir_->insts(), FormatterChunks::None) {
  if (use_dump_sem_ir_ranges_) {
    ComputeNodeParents();
  }

  // Reserve space for parents. There will be more content, but we don't try to
  // guess how much.
  size_t reserve_chunks = scope_label_chunks_.size();
  for (auto [_, insts] : GetTentativeScopes(*sem_ir_)) {
    reserve_chunks += insts.size();
  }
  chunks_.Reserve(reserve_chunks);

  // Create parent chunks for scopes.
  for (auto& chunk : scope_label_chunks_) {
    chunk = chunks_.AddParent();
  }

  // Create parent chunks for the tentative instructions.
  for (auto [scope_id, insts] : GetTentativeScopes(*sem_ir_)) {
    auto scope_chunk = scope_label_chunks_[static_cast<size_t>(scope_id)];
    for (auto inst_id : insts) {
      // Instructions are "parents" of their scopes because if any instruction
      // is printed, the label is also printed.
      tentative_inst_chunks_.Set(inst_id, chunks_.AddParent(scope_chunk));
    }
  }

  CARBON_CHECK(chunks_.size() == reserve_chunks);

  // Prepare to add content.
  chunks_.StartContent();
}

auto Formatter::Format() -> void {
  out() << "--- " << sem_ir_->filename() << "\n";

  for (auto [scope_id, insts] : GetTentativeScopes(*sem_ir_)) {
    FormatTopLevelScope(scope_id, insts);
  }

  FormatTopLevelScope(
      InstNamer::ScopeId::File,
      sem_ir_->inst_blocks().GetOrEmpty(sem_ir_->top_inst_block_id()));

  for (const auto& [id, interface] : sem_ir_->interfaces().enumerate()) {
    FormatInterface(id, interface);
  }

  for (const auto& [id, constraint] :
       sem_ir_->named_constraints().enumerate()) {
    FormatNamedConstraint(id, constraint);
  }

  for (const auto& [id, require] : sem_ir_->require_impls().enumerate()) {
    FormatRequireImpls(id, require);
  }

  for (const auto& [id, impl] : sem_ir_->impls().enumerate()) {
    FormatImpl(id, impl);
  }

  for (const auto& [id, class_info] : sem_ir_->classes().enumerate()) {
    FormatClass(id, class_info);
  }

  for (const auto& [id, vtable] : sem_ir_->vtables().enumerate()) {
    FormatVtable(id, vtable);
  }

  for (const auto& [id, function] : sem_ir_->functions().enumerate()) {
    FormatFunction(id, function);
  }

  for (const auto& [id, specific] : sem_ir_->specifics().enumerate()) {
    FormatSpecific(id, specific);
  }

  out() << "\n";
}

auto Formatter::ComputeNodeParents() -> void {
  CARBON_CHECK(!node_parents_);
  node_parents_ = NodeParentStore::MakeWithExplicitSize(
      sem_ir_->parse_tree().size(), Parse::NodeId::None);
  for (auto n : sem_ir_->parse_tree().postorder()) {
    for (auto child : get_tree_and_subtrees_().children(n)) {
      node_parents_->Set(child, n);
    }
  }
}

auto Formatter::Write(llvm::raw_ostream& out) -> void { chunks_.Write(out); }

auto Formatter::ShouldIncludeInstByIR(InstId inst_id) -> bool {
  const auto* import_ir = GetCanonicalFileAndInstId(sem_ir_, inst_id).first;
  return include_ir_in_dumps_->Get(import_ir->check_ir_id());
}

// Returns true for a `DefinitionStart` node.
static auto IsDefinitionStart(Parse::NodeKind node_kind) -> bool {
  switch (node_kind) {
    case Parse::NodeKind::BuiltinFunctionDefinitionStart:
    case Parse::NodeKind::ChoiceDefinitionStart:
    case Parse::NodeKind::ClassDefinitionStart:
    case Parse::NodeKind::FunctionDefinitionStart:
    case Parse::NodeKind::ImplDefinitionStart:
    case Parse::NodeKind::InterfaceDefinitionStart:
    case Parse::NodeKind::NamedConstraintDefinitionStart:
      return true;
    default:
      return false;
  }
}

auto Formatter::ShouldFormatEntity(InstId decl_id) -> bool {
  if (!decl_id.has_value()) {
    return true;
  }
  if (!ShouldIncludeInstByIR(decl_id)) {
    return false;
  }

  if (!use_dump_sem_ir_ranges_) {
    return true;
  }

  // When there are dump ranges, ignore imported instructions.
  auto loc_id = sem_ir_->insts().GetCanonicalLocId(decl_id);
  if (loc_id.kind() != LocId::Kind::NodeId) {
    return false;
  }

  const auto& tree_and_subtrees = get_tree_and_subtrees_();

  // This takes the earliest token from either the node or its first postorder
  // child. The first postorder child isn't necessarily the earliest token in
  // the subtree (for example, it can miss modifiers), but finding the earliest
  // token requires walking *all* children, whereas this approach is
  // constant-time.
  auto begin_node_id = *tree_and_subtrees.postorder(loc_id.node_id()).begin();

  // Non-defining declarations will be associated with a `Decl` node.
  // Definitions will have a `DefinitionStart` for which we can use the parent
  // to find the `Definition`, giving a range that includes the definition's
  // body.
  auto end_node_id = loc_id.node_id();
  if (IsDefinitionStart(sem_ir_->parse_tree().node_kind(end_node_id))) {
    end_node_id = node_parents_->Get(end_node_id);
  }

  Lex::InclusiveTokenRange range = {
      .begin = sem_ir_->parse_tree().node_token(begin_node_id),
      .end = sem_ir_->parse_tree().node_token(end_node_id)};
  return sem_ir_->parse_tree().tokens().OverlapsWithDumpSemIRRange(range);
}

auto Formatter::ShouldFormatEntity(const EntityWithParamsBase& entity) -> bool {
  return ShouldFormatEntity(entity.latest_decl_id());
}

auto Formatter::ShouldFormatInst(InstId inst_id) -> bool {
  if (!use_dump_sem_ir_ranges_) {
    return true;
  }

  // When there are dump ranges, ignore imported instructions.
  auto loc_id = sem_ir_->insts().GetCanonicalLocId(inst_id);
  if (loc_id.kind() != LocId::Kind::NodeId) {
    return false;
  }

  auto token = sem_ir_->parse_tree().node_token(loc_id.node_id());
  return sem_ir_->parse_tree().tokens().OverlapsWithDumpSemIRRange(
      Lex::InclusiveTokenRange{.begin = token, .end = token});
}

auto Formatter::OpenBrace() -> void {
  // Put the constant value of an instruction before any braced block, rather
  // than at the end.
  FormatPendingConstantValue(AddSpace::After);

  // Put the imported-from library name before the definition of the entity.
  FormatPendingImportedFrom(AddSpace::After);

  out() << '{';
  indent_ += 2;
  after_open_brace_ = true;
}

auto Formatter::CloseBrace() -> void {
  indent_ -= 2;
  if (!after_open_brace_) {
    Indent();
  }
  out() << '}';
  after_open_brace_ = false;
}

auto Formatter::Semicolon() -> void {
  FormatPendingImportedFrom(AddSpace::Before);
  out() << ';';
}

auto Formatter::Indent(int offset) -> void {
  if (after_open_brace_) {
    out() << '\n';
    after_open_brace_ = false;
  }
  out().indent(indent_ + offset);
}

auto Formatter::IndentLabel() -> void {
  CARBON_CHECK(indent_ >= 2);
  if (!after_open_brace_) {
    out() << '\n';
  }
  Indent(-2);
}

auto Formatter::FormatTopLevelScope(InstNamer::ScopeId scope_id,
                                    llvm::ArrayRef<InstId> block) -> void {
  if (block.empty()) {
    return;
  }

  llvm::SaveAndRestore scope(scope_, scope_id);
  auto scope_chunk = scope_label_chunks_[static_cast<size_t>(scope_id)];

  chunks_.FormatChildContent(scope_chunk, [&] {
    // Note, we don't use OpenBrace() / CloseBrace() here because we always want
    // a newline to avoid misformatting if the first instruction is omitted.
    out() << "\n" << inst_namer_.GetScopeName(scope_id) << " {\n";
  });

  indent_ += 2;
  for (const InstId inst_id : block) {
    // Format instructions when needed, but do nothing for elided entries;
    // unlike normal code blocks, scopes are non-sequential so skipped
    // instructions are assumed to be uninteresting.
    if (scope_id == InstNamer::ScopeId::File) {
      // Applies range-based filtering of instructions.
      if (!ShouldFormatInst(inst_id)) {
        continue;
      }

      FormatInst(inst_id);
      // Include the `file` scope label directly here.
      chunks_.AppendChildToCurrentParent(scope_chunk);
    } else {
      // Other scopes format each instruction in its own chunk, to support
      // tentative formatting.
      chunks_.FormatChildContent(tentative_inst_chunks_.Get(inst_id),
                                 [&] { FormatInst(inst_id); });
    }
  }
  indent_ -= 2;

  chunks_.FormatChildContent(scope_chunk, [&] { out() << "}\n"; });
}

auto Formatter::FormatClass(ClassId id, const Class& class_info) -> void {
  if (!ShouldFormatEntity(class_info)) {
    return;
  }

  PrepareToFormatDecl(class_info.first_owning_decl_id);
  FormatEntityStart("class", class_info, id);

  llvm::SaveAndRestore class_scope(scope_, inst_namer_.GetScopeFor(id));

  if (class_info.scope_id.has_value()) {
    out() << ' ';
    OpenBrace();
    FormatCodeBlock(class_info.body_block_id);
    Indent();
    out() << "complete_type_witness = ";
    FormatName(class_info.complete_type_witness_id);
    out() << "\n";
    if (class_info.vtable_decl_id.has_value()) {
      Indent();
      out() << "vtable_decl = ";
      FormatName(class_info.vtable_decl_id);
      out() << "\n";
    }

    FormatNameScope(class_info.scope_id, "!members:\n");
    CloseBrace();
  } else {
    Semicolon();
  }
  out() << '\n';

  FormatEntityEnd(class_info.generic_id);
}

auto Formatter::FormatVtable(VtableId id, const Vtable& vtable_info) -> void {
  out() << '\n';
  Indent();
  out() << "vtable ";
  FormatName(id);
  out() << ' ';
  OpenBrace();
  for (auto function_id :
       sem_ir_->inst_blocks().Get(vtable_info.virtual_functions_id)) {
    Indent();
    FormatArg(function_id);
    out() << '\n';
  }
  CloseBrace();
  out() << '\n';
}

auto Formatter::FormatInterface(InterfaceId id, const Interface& interface_info)
    -> void {
  if (!ShouldFormatEntity(interface_info)) {
    return;
  }

  PrepareToFormatDecl(interface_info.first_owning_decl_id);
  FormatEntityStart("interface", interface_info, id);

  llvm::SaveAndRestore interface_scope(scope_, inst_namer_.GetScopeFor(id));

  if (interface_info.is_complete()) {
    out() << ' ';
    OpenBrace();
    FormatCodeBlock(interface_info.body_block_without_self_id);

    bool body_block_empty =
        sem_ir_->inst_blocks()
            .GetOrEmpty(interface_info.body_block_with_self_id)
            .empty();
    if (!body_block_empty) {
      IndentLabel();
      out() << "!with Self:\n";

      llvm::SaveAndRestore with_self_scope(
          scope_, inst_namer_.GetScopeFor(InstNamer::InterfaceWithSelfId{id}));
      FormatCodeBlock(interface_info.body_block_with_self_id);
    }

    // Always include the !members without self label because we always list the
    // witness in this section.
    IndentLabel();
    out() << "!members:\n";

    FormatNameScope(interface_info.scope_without_self_id);
    FormatNameScope(interface_info.scope_with_self_id);

    Indent();
    out() << "witness = ";
    FormatArg(interface_info.associated_entities_id);
    out() << "\n";

    FormatRequireImplsBlock(interface_info.require_impls_block_id);

    CloseBrace();
  } else {
    Semicolon();
  }
  out() << '\n';

  FormatEntityEnd(interface_info.generic_id);
}

auto Formatter::FormatNamedConstraint(NamedConstraintId id,
                                      const NamedConstraint& constraint_info)
    -> void {
  if (!ShouldFormatEntity(constraint_info)) {
    return;
  }

  PrepareToFormatDecl(constraint_info.first_owning_decl_id);
  FormatEntityStart("constraint", constraint_info, id);

  llvm::SaveAndRestore constraint_scope(scope_, inst_namer_.GetScopeFor(id));

  if (constraint_info.is_complete()) {
    out() << ' ';
    OpenBrace();
    FormatCodeBlock(constraint_info.body_block_without_self_id);

    bool body_block_empty =
        sem_ir_->inst_blocks()
            .GetOrEmpty(constraint_info.body_block_with_self_id)
            .empty();
    if (!body_block_empty) {
      IndentLabel();
      out() << "!with Self:\n";

      llvm::SaveAndRestore with_self_scope(
          scope_,
          inst_namer_.GetScopeFor(InstNamer::NamedConstraintWithSelfId{id}));
      FormatCodeBlock(constraint_info.body_block_with_self_id);
    }

    // Always include the !members label because we always list the witness in
    // this section.
    IndentLabel();
    out() << "!members:\n";
    FormatNameScope(constraint_info.scope_without_self_id);
    FormatNameScope(constraint_info.scope_with_self_id);

    FormatRequireImplsBlock(constraint_info.require_impls_block_id);

    CloseBrace();
  } else {
    Semicolon();
  }
  out() << '\n';

  FormatEntityEnd(constraint_info.generic_id);
}

auto Formatter::FormatRequireImpls(RequireImplsId /*id*/,
                                   const RequireImpls& require) -> void {
  if (!ShouldFormatEntity(require.decl_id)) {
    return;
  }

  PrepareToFormatDecl(require.decl_id);
  FormatGenericStart("require", require.generic_id);
  FormatGenericEnd();
}

auto Formatter::FormatImpl(ImplId id, const Impl& impl_info) -> void {
  if (!ShouldFormatEntity(impl_info)) {
    return;
  }

  PrepareToFormatDecl(impl_info.first_owning_decl_id);
  FormatEntityStart("impl", impl_info, id);

  llvm::SaveAndRestore impl_scope(scope_, inst_namer_.GetScopeFor(id));

  out() << ": ";
  FormatName(impl_info.self_id);
  out() << " as ";
  FormatName(impl_info.constraint_id);

  if (impl_info.is_complete()) {
    out() << ' ';
    OpenBrace();
    FormatCodeBlock(impl_info.body_block_id);
    FormatCodeBlock(impl_info.witness_block_id);

    // Print the !members label even if the name scope is empty because we
    // always list the witness in this section.
    IndentLabel();
    out() << "!members:\n";
    if (impl_info.scope_id.has_value()) {
      FormatNameScope(impl_info.scope_id);
    }

    Indent();
    out() << "witness = ";
    FormatArg(impl_info.witness_id);
    out() << "\n";

    CloseBrace();
  } else {
    Semicolon();
  }
  out() << '\n';

  FormatEntityEnd(impl_info.generic_id);
}

auto Formatter::FormatFunction(FunctionId id, const Function& fn) -> void {
  if (!ShouldFormatEntity(fn)) {
    return;
  }

  std::string function_start;
  switch (fn.virtual_modifier) {
    case FunctionFields::VirtualModifier::Virtual:
      function_start += "virtual ";
      break;
    case FunctionFields::VirtualModifier::Abstract:
      function_start += "abstract ";
      break;
    case FunctionFields::VirtualModifier::Override:
      function_start += "override ";
      break;
    case FunctionFields::VirtualModifier::None:
      break;
  }
  if (fn.is_extern) {
    function_start += "extern ";
  }
  function_start += "fn";
  PrepareToFormatDecl(fn.first_owning_decl_id);
  FormatEntityStart(function_start, fn, id);

  llvm::SaveAndRestore function_scope(scope_, inst_namer_.GetScopeFor(id));

  FormatFunctionSignature(fn.call_params_id,
                          fn.call_param_ranges.return_begin(),
                          fn.GetDeclaredReturnForm(*sem_ir_));

  if (fn.builtin_function_kind() != BuiltinFunctionKind::None) {
    out() << " = \""
          << FormatEscaped(fn.builtin_function_kind().name(),
                           /*use_hex_escapes=*/true)
          << "\"";
  }
  if (fn.thunk_id().has_value()) {
    out() << " [thunk ";
    const auto& thunk_info = sem_ir_->thunks().Get(fn.thunk_id());
    FormatArg(thunk_info.callee_id);
    if (thunk_info.signature_id.has_value()) {
      out() << " for ";
      FormatName(sem_ir_->functions()
                     .Get(thunk_info.signature_id)
                     .first_owning_decl_id);
      if (thunk_info.specific_id.has_value()) {
        out() << ", ";
        FormatName(thunk_info.specific_id);
      }
    }
    out() << "]";
  }

  if (!fn.body_block_ids.empty()) {
    out() << ' ';
    OpenBrace();

    for (auto block_id : fn.body_block_ids) {
      IndentLabel();
      FormatLabel(block_id);
      out() << ":\n";

      FormatCodeBlock(block_id);
    }

    CloseBrace();
  } else {
    Semicolon();
  }
  out() << '\n';

  FormatEntityEnd(fn.generic_id);
}

auto Formatter::FormatSpecificRegion(const Generic& generic,
                                     const Specific& specific,
                                     GenericInstIndex::Region region,
                                     llvm::StringRef region_name) -> void {
  if (!specific.GetValueBlock(region).has_value()) {
    return;
  }

  if (!region_name.empty()) {
    IndentLabel();
    out() << "!" << region_name << ":\n";
  }
  for (auto [generic_inst_id, specific_inst_id] : llvm::zip_longest(
           sem_ir_->inst_blocks().GetOrEmpty(generic.GetEvalBlock(region)),
           sem_ir_->inst_blocks().GetOrEmpty(specific.GetValueBlock(region)))) {
    Indent();
    if (generic_inst_id) {
      FormatName(*generic_inst_id);
    } else {
      out() << "<missing>";
    }
    out() << " => ";
    if (specific_inst_id) {
      FormatName(*specific_inst_id);
    } else {
      out() << "<missing>";
    }
    out() << "\n";
  }
}

auto Formatter::FormatSpecific(SpecificId id, const Specific& specific)
    -> void {
  const auto& generic = sem_ir_->generics().Get(specific.generic_id);
  if (!ShouldFormatEntity(generic.decl_id)) {
    // Omit specifics if we also omitted the generic.
    return;
  }

  if (specific.IsUnresolved()) {
    // Omit specifics that were never resolved. Such specifics exist only to
    // track the way the arguments were spelled, and that information is
    // conveyed entirely by the name of the specific. These specifics may also
    // not be referenced by any SemIR that we format, so including them adds
    // clutter and possibly emits references to instructions we didn't name.
    return;
  }

  llvm::SaveAndRestore generic_scope(
      scope_, inst_namer_.GetScopeFor(specific.generic_id));

  out() << "\n";

  out() << "specific ";
  FormatName(id);
  out() << " ";

  OpenBrace();
  FormatSpecificRegion(generic, specific, GenericInstIndex::Region::Declaration,
                       "");
  FormatSpecificRegion(generic, specific, GenericInstIndex::Region::Definition,
                       "definition");
  CloseBrace();

  out() << "\n";
}

auto Formatter::PrepareToFormatDecl(InstId first_owning_decl_id) -> void {
  // If this decl was imported from a different IR, annotate the name of
  // that IR in the output before the `{` or `;`.
  if (first_owning_decl_id.has_value()) {
    auto import_ir_inst_id =
        sem_ir_->insts().GetImportSource(first_owning_decl_id);
    if (import_ir_inst_id.has_value()) {
      auto import_ir_id =
          sem_ir_->import_ir_insts().Get(import_ir_inst_id).ir_id();
      if (const auto* import_file =
              sem_ir_->import_irs().Get(import_ir_id).sem_ir) {
        pending_imported_from_ = import_file->filename();
      }
    }
  }
}

auto Formatter::FormatGenericStart(llvm::StringRef entity_kind,
                                   GenericId generic_id) -> void {
  const auto& generic = sem_ir_->generics().Get(generic_id);
  out() << "\n";
  Indent();
  out() << "generic " << entity_kind << " ";
  FormatName(generic_id);

  llvm::SaveAndRestore generic_scope(scope_,
                                     inst_namer_.GetScopeFor(generic_id));

  FormatParamList(generic.bindings_id);

  out() << " ";
  OpenBrace();
  FormatCodeBlock(generic.decl_block_id);
  if (generic.definition_block_id.has_value()) {
    IndentLabel();
    out() << "!definition:\n";
    FormatCodeBlock(generic.definition_block_id);
  }
}

auto Formatter::FormatEntityEnd(GenericId generic_id) -> void {
  if (generic_id.has_value()) {
    FormatGenericEnd();
  }
}

auto Formatter::FormatGenericEnd() -> void {
  CloseBrace();
  out() << '\n';
}

auto Formatter::FormatFunctionSignature(InstBlockId params_id,
                                        SemIR::CallParamIndex return_begin,
                                        InstId return_form_id) -> void {
  if (!params_id.has_value()) {
    // TODO: This happens for imported functions, for which we don't currently
    // import the call parameters list.
    return;
  }

  auto params = sem_ir_->inst_blocks().Get(params_id);

  out() << "(";
  llvm::ListSeparator sep;
  int i = 0;
  for (auto param_id : params) {
    if (return_begin.has_value() && i >= return_begin.index) {
      break;
    }

    out() << sep;
    if (!param_id.has_value()) {
      out() << "invalid";
      continue;
    }
    if (sem_ir_->insts().Is<OutParam>(param_id)) {
      out() << "<unexpected out> ";
    }
    FormatNameAndForm(param_id, sem_ir_->insts().Get(param_id));
    ++i;
  }

  out() << ")";

  if (return_form_id.has_value()) {
    out() << " -> ";
    auto return_form = sem_ir_->insts().Get(return_form_id);
    CARBON_KIND_SWITCH(return_form) {
      case CARBON_KIND(InitForm _): {
        out() << "out ";
        FormatName(params[i]);
        out() << ": ";
        FormatTypeOfInst(params[i]);
        ++i;
        break;
      }
      case CARBON_KIND(RefForm ref_form): {
        out() << "ref ";
        FormatInstAsType(ref_form.type_component_inst_id);
        break;
      }
      case CARBON_KIND(ValueForm val_form): {
        out() << "val ";
        FormatInstAsType(val_form.type_component_inst_id);
        break;
      }
      case CARBON_KIND(ErrorInst _): {
        FormatInstAsType(return_form_id);
        break;
      }
      case CARBON_KIND(SpliceInst splice): {
        out() << "out ";
        FormatName(params[i]);
        out() << ":? ";
        // A form isn't a type, but it's close enough for formatting purposes.
        FormatInstAsType(splice.inst_id);
        ++i;
        break;
      }
      default:
        CARBON_FATAL("Unexpected inst kind: {0}", return_form);
    }
  }
  CARBON_CHECK(i == static_cast<int>(params.size()),
               "`return_begin` and `return_form_id` imply different numbers of "
               "return params.");
}

auto Formatter::FormatCodeBlock(InstBlockId block_id) -> void {
  bool elided = false;
  for (const InstId inst_id : sem_ir_->inst_blocks().GetOrEmpty(block_id)) {
    if (ShouldFormatInst(inst_id)) {
      FormatInst(inst_id);
      elided = false;
    } else if (!elided) {
      // When formatting a block, leave a hint that instructions were elided.
      Indent();
      out() << "<elided>\n";
      elided = true;
    }
  }
}

auto Formatter::FormatTrailingBlock(InstBlockId block_id) -> void {
  out() << ' ';
  OpenBrace();
  FormatCodeBlock(block_id);
  CloseBrace();
}

auto Formatter::FormatNameScope(NameScopeId id, llvm::StringRef label) -> void {
  const auto& scope = sem_ir_->name_scopes().Get(id);

  if (scope.entries().empty() && scope.extended_scopes().empty() &&
      scope.import_ir_scopes().empty() && !scope.is_cpp_scope() &&
      !scope.has_error()) {
    // Name scope is empty.
    return;
  }

  if (!label.empty()) {
    IndentLabel();
    out() << label;
  }

  for (auto [name_id, result] : scope.entries()) {
    Indent();
    out() << ".";
    FormatName(name_id);
    switch (result.access_kind()) {
      case AccessKind::Public:
        break;
      case AccessKind::Protected:
        out() << " [protected]";
        break;
      case AccessKind::Private:
        out() << " [private]";
        break;
    }
    out() << " = ";
    if (result.is_poisoned()) {
      out() << "<poisoned>";
    } else {
      FormatName(result.is_found() ? result.target_inst_id() : InstId::None);
    }
    out() << "\n";
  }

  for (auto extended_scope_id : scope.extended_scopes()) {
    Indent();
    out() << "extend ";
    FormatName(extended_scope_id);
    out() << "\n";
  }

  // This is used to cluster all "Core//prelude/..." imports, but not
  // "Core//prelude" itself. This avoids unrelated churn in test files when we
  // add or remove an unused prelude file, but is intended to still show the
  // existence of indirect imports.
  bool has_prelude_components = false;
  for (auto [import_ir_id, unused] : scope.import_ir_scopes()) {
    auto label = GetImportIRLabel(import_ir_id);
    if (label.starts_with("Core//prelude/")) {
      if (has_prelude_components) {
        // Only print the existence once.
        continue;
      } else {
        has_prelude_components = true;
        label = "Core//prelude/...";
      }
    }
    Indent();
    out() << "import " << label << "\n";
  }

  if (scope.is_cpp_scope()) {
    Indent();
    out() << "import Cpp//...\n";
  }

  if (scope.has_error()) {
    Indent();
    out() << "has_error\n";
  }
}

auto Formatter::FormatInst(InstId inst_id) -> void {
  if (!inst_id.has_value()) {
    Indent();
    out() << "none\n";
    return;
  }

  if (!in_terminator_sequence_) {
    Indent();
  }

  auto inst = sem_ir_->insts().GetWithAttachedType(inst_id);
  CARBON_KIND_SWITCH(inst) {
    case CARBON_KIND(Branch branch): {
      out() << Branch::Kind.ir_name() << " ";
      FormatLabel(branch.target_id);
      out() << "\n";
      in_terminator_sequence_ = false;
      return;
    }
    case CARBON_KIND(BranchIf branch_if): {
      out() << "if ";
      FormatName(branch_if.cond_id);
      out() << " " << Branch::Kind.ir_name() << " ";
      FormatLabel(branch_if.target_id);
      out() << " else ";
      in_terminator_sequence_ = true;
      return;
    }
    case CARBON_KIND(BranchWithArg branch_with_arg): {
      out() << BranchWithArg::Kind.ir_name() << " ";
      FormatLabel(branch_with_arg.target_id);
      out() << "(";
      FormatName(branch_with_arg.arg_id);
      out() << ")\n";
      in_terminator_sequence_ = false;
      return;
    }
    default: {
      FormatInstLhs(inst_id, inst);
      out() << inst.kind().ir_name();

      // Add constants for everything except `ImportRefUnloaded`.
      if (!inst.Is<ImportRefUnloaded>()) {
        pending_constant_value_ =
            sem_ir_->constant_values().GetAttached(inst_id);
        pending_constant_value_is_self_ =
            sem_ir_->constant_values().GetInstIdIfValid(
                pending_constant_value_) == inst_id;
      }

      FormatInstRhs(inst);
      // This usually prints the constant, but when `FormatInstRhs` prints it
      // first (or for `ImportRefUnloaded`), this does nothing.
      FormatPendingConstantValue(AddSpace::Before);
      out() << "\n";
      return;
    }
  }
}

auto Formatter::FormatPendingImportedFrom(AddSpace space_where) -> void {
  if (pending_imported_from_.empty()) {
    return;
  }

  if (space_where == AddSpace::Before) {
    out() << ' ';
  }
  out() << "[from \"" << FormatEscaped(pending_imported_from_) << "\"]";
  if (space_where == AddSpace::After) {
    out() << ' ';
  }
  pending_imported_from_ = llvm::StringRef();
}

auto Formatter::FormatPendingConstantValue(AddSpace space_where) -> void {
  if (pending_constant_value_ == ConstantId::NotConstant) {
    return;
  }

  if (space_where == AddSpace::Before) {
    out() << ' ';
  }
  out() << '[';
  if (pending_constant_value_.has_value()) {
    switch (sem_ir_->constant_values().GetDependence(pending_constant_value_)) {
      case ConstantDependence::None:
        out() << "concrete";
        break;
      case ConstantDependence::PeriodSelf:
        out() << "symbolic_self";
        break;
      // TODO: Consider renaming this. This will cause a lot of SemIR churn.
      case ConstantDependence::Checked:
        out() << "symbolic";
        break;
      case ConstantDependence::Template:
        out() << "template";
        break;
    }
    if (!pending_constant_value_is_self_) {
      out() << " = ";
      FormatConstant(pending_constant_value_);
    }
  } else {
    out() << pending_constant_value_;
  }
  out() << ']';
  if (space_where == AddSpace::After) {
    out() << ' ';
  }
  pending_constant_value_ = ConstantId::NotConstant;
}

auto Formatter::FormatInstLhs(InstId inst_id, Inst inst) -> void {
  // Every typed instruction is named, and there are some untyped instructions
  // that have names (such as `ImportRefUnloaded`). When there's a typed
  // instruction with no name, it means an instruction is incorrectly not named
  // -- but should be printed as such.
  bool has_name = inst_namer_.has_name(inst_id);
  if (!has_name && !inst.kind().has_type()) {
    return;
  }

  FormatNameAndForm(inst_id, inst);

  out() << " = ";
}

auto Formatter::FormatNameAndForm(InstId inst_id, Inst inst) -> void {
  FormatName(inst_id);

  if (inst.kind().has_type()) {
    out() << ": ";
    switch (GetExprCategory(*sem_ir_, inst_id)) {
      case ExprCategory::NotExpr:
      case ExprCategory::Error:
      case ExprCategory::Value:
      case ExprCategory::Pattern:
      case ExprCategory::Mixed:
      case ExprCategory::RefTagged:
      case ExprCategory::Dependent:
        FormatTypeOfInst(inst_id);
        break;
      case ExprCategory::DurableRef:
      case ExprCategory::EphemeralRef:
        out() << "ref ";
        FormatTypeOfInst(inst_id);
        break;
      case ExprCategory::InPlaceInitializing:
      case ExprCategory::ReprInitializing: {
        out() << "init ";
        FormatTypeOfInst(inst_id);
        auto init_target_id = FindStorageArgForInitializer(
            *sem_ir_, inst_id, /*allow_transitive=*/false);
        FormatReturnSlotArg(init_target_id);
        break;
      }
    }
  }
}

auto Formatter::FormatInstArgAndKind(IdAndKind arg_and_kind) -> void {
  arg_and_kind.Dispatch<void>([this]<typename IdT>(IdT arg) {
    if constexpr (requires { FormatArg(arg); }) {
      FormatArg(arg);
    } else if constexpr (std::is_same_v<IdT, IdAndKind::NoneType>) {
      // Do nothing
    } else {
      CARBON_FATAL("Missing FormatArg for {0}", typeid(IdT).name());
    }
  });
}

auto Formatter::FormatInstRhs(Inst inst) -> void {
  CARBON_KIND_SWITCH(inst) {
    case CARBON_KIND_ANY(AnyAggregateInit, init): {
      FormatArgs(init.elements_id);
      return;
    }

    case CARBON_KIND_ANY(AnyImportRef, import_ref): {
      FormatImportRefRhs(import_ref);
      return;
    }

    case CARBON_KIND_ANY(AnyParam, param): {
      FormatArgs(param.index);
      // Omit pretty_name because it's an implementation detail of
      // pretty-printing.
      return;
    }

    case CARBON_KIND_ANY(AnyLeafParamPattern, _): {
      // Omit pretty_name because it's an implementation detail of
      // pretty-printing.
      return;
    }

    case CARBON_KIND(VarParamPattern param): {
      FormatArgs(param.subpattern_id);
      // Omit pretty_name because it's an implementation detail of
      // pretty-printing.
      return;
    }

    case CARBON_KIND(AssociatedConstantDecl decl): {
      FormatArgs(decl.assoc_const_id);
      llvm::SaveAndRestore scope(scope_,
                                 inst_namer_.GetScopeFor(decl.assoc_const_id));
      FormatTrailingBlock(decl.decl_block_id);
      return;
    }

    case CARBON_KIND(SymbolicBinding bind): {
      // A SymbolicBinding with no value is a purely symbolic binding, such as
      // the `Self` in an interface. Don't print out `none` for the value.
      if (bind.value_id.has_value()) {
        FormatArgs(bind.entity_name_id, bind.value_id);
      } else {
        FormatArgs(bind.entity_name_id);
      }
      return;
    }

    case CARBON_KIND(BlockArg block): {
      out() << " ";
      FormatLabel(block.block_id);
      return;
    }

    case CARBON_KIND(Call call): {
      FormatCallRhs(call);
      return;
    }

    case CARBON_KIND(ClassDecl decl): {
      FormatDeclRhs(decl.class_id,
                    sem_ir_->classes().Get(decl.class_id).pattern_block_id,
                    decl.decl_block_id);
      return;
    }

    case CARBON_KIND(CppTemplateNameType type): {
      // Omit the Clang declaration. We don't have a good way to format it, and
      // the entity name should suffice to identify the template.
      FormatArgs(type.name_id);
      return;
    }

    case CARBON_KIND(CustomLayoutType type): {
      out() << " {";
      auto layout = sem_ir_->custom_layouts().Get(type.layout_id);
      out() << "size=" << layout[CustomLayoutId::SizeIndex]
            << ", align=" << layout[CustomLayoutId::AlignIndex];
      for (auto [field, offset] : llvm::zip_equal(
               sem_ir_->struct_type_fields().Get(type.fields_id),
               layout.drop_front(CustomLayoutId::FirstFieldIndex))) {
        out() << ", .";
        FormatName(field.name_id);
        out() << "@" << offset << ": ";
        FormatInstAsType(field.type_inst_id);
      }
      out() << "}";
      return;
    }

    case CARBON_KIND(FloatValue value): {
      llvm::SmallVector<char, 16> buffer;
      sem_ir_->floats().Get(value.float_id).toString(buffer);
      out() << " " << buffer;
      return;
    }

    case CARBON_KIND(FunctionDecl decl): {
      FormatDeclRhs(decl.function_id,
                    sem_ir_->functions().Get(decl.function_id).pattern_block_id,
                    decl.decl_block_id);
      return;
    }

    case ImportCppDecl::Kind: {
      FormatImportCppDeclRhs();
      return;
    }

    case CARBON_KIND(ImplDecl decl): {
      FormatDeclRhs(decl.impl_id,
                    sem_ir_->impls().Get(decl.impl_id).pattern_block_id,
                    decl.decl_block_id);
      return;
    }

    case CARBON_KIND(InPlaceInit init): {
      FormatArgs(init.src_id);
      return;
    }

    case CARBON_KIND(InstValue inst): {
      out() << ' ';
      OpenBrace();
      // TODO: Should we use a more compact representation in the case where the
      // inst is a SpliceBlock?
      FormatInst(inst.inst_id);
      CloseBrace();
      return;
    }

    case CARBON_KIND(InterfaceDecl decl): {
      FormatDeclRhs(
          decl.interface_id,
          sem_ir_->interfaces().Get(decl.interface_id).pattern_block_id,
          decl.decl_block_id);
      return;
    }

    case CARBON_KIND(IntValue value): {
      out() << " ";
      sem_ir_->ints()
          .Get(value.int_id)
          .print(out(), sem_ir_->types().IsSignedInt(value.type_id));
      return;
    }

    case CARBON_KIND(NameBindingDecl name): {
      FormatTrailingBlock(name.pattern_block_id);
      return;
    }

    case CARBON_KIND(NamedConstraintDecl decl): {
      FormatDeclRhs(decl.named_constraint_id,
                    sem_ir_->named_constraints()
                        .Get(decl.named_constraint_id)
                        .pattern_block_id,
                    decl.decl_block_id);
      return;
    }

    case CARBON_KIND(Namespace ns): {
      if (ns.import_id.has_value()) {
        FormatArgs(ns.import_id, ns.name_scope_id);
      } else {
        FormatArgs(ns.name_scope_id);
      }
      return;
    }

    case CARBON_KIND(RequireImplsDecl decl): {
      FormatArgs(decl.require_impls_id);
      llvm::SaveAndRestore scope(
          scope_, inst_namer_.GetScopeFor(decl.require_impls_id));
      FormatRequireImpls(decl.require_impls_id);
      FormatTrailingBlock(decl.decl_block_id);
      return;
    }

    case CARBON_KIND(ReturnExpr ret): {
      FormatArgs(ret.expr_id);
      if (ret.dest_id.has_value()) {
        FormatReturnSlotArg(ret.dest_id);
      }
      return;
    }

    case CARBON_KIND(ReturnSlot ret): {
      // Omit inst.type_inst_id because it's not semantically significant.
      FormatArgs(ret.storage_id);
      return;
    }

    case CARBON_KIND(SpliceBlock splice): {
      FormatArgs(splice.result_id);
      FormatTrailingBlock(splice.block_id);
      return;
    }

    case CARBON_KIND(StructType struct_type): {
      out() << " {";
      llvm::ListSeparator sep;
      for (auto field :
           sem_ir_->struct_type_fields().Get(struct_type.fields_id)) {
        out() << sep << ".";
        FormatName(field.name_id);
        out() << ": ";
        FormatInstAsType(field.type_inst_id);
      }
      out() << "}";
      return;
    }

    case CARBON_KIND(WhereExpr where): {
      FormatTrailingBlock(where.requirements_id);
      return;
    }

    default:
      FormatInstRhsDefault(inst);
      return;
  }
}

auto Formatter::FormatInstRhsDefault(Inst inst) -> void {
  auto arg0 = inst.arg0_and_kind();
  if (arg0.kind() == IdKind::None) {
    return;
  }
  out() << " ";
  FormatInstArgAndKind(arg0);

  auto arg1 = inst.arg1_and_kind();
  if (arg1.kind() == IdKind::None) {
    return;
  }

  // Several instructions have a second operand that's a specific ID. We
  // don't include it in the argument list if there is no corresponding
  // specific, that is, when we're not in a generic context.
  if (auto arg1_specific_id = arg1.TryAs<SpecificId>();
      arg1_specific_id && !arg1_specific_id->has_value()) {
    return;
  }
  // Similarly, instructions that have a `DestInstId` as the second operand
  // typically use it for the output argument, so we omit it because it should
  // already be part of the inst's formatted form expression.
  if (arg1.kind() == IdKind::For<DestInstId>) {
    return;
  }
  out() << ", ";
  FormatInstArgAndKind(arg1);
}

auto Formatter::FormatCallRhs(Call inst) -> void {
  out() << " ";
  FormatArg(inst.callee_id);

  if (!inst.args_id.has_value()) {
    out() << "(<none>)";
    return;
  }

  llvm::ArrayRef<InstId> args = sem_ir_->inst_blocks().Get(inst.args_id);

  // If there are return arguments, don't print them here, because it's printed
  // on the LHS.
  auto explicit_end = SemIR::CallParamIndex::None;
  auto callee = GetCallee(*sem_ir_, inst.callee_id);
  if (auto* callee_function = std::get_if<CalleeFunction>(&callee)) {
    explicit_end = sem_ir_->functions()
                       .Get(callee_function->function_id)
                       .call_param_ranges.explicit_end();
  }

  llvm::ListSeparator sep;
  out() << '(';
  for (auto [i, inst_id] : llvm::enumerate(args)) {
    if (explicit_end.has_value() && static_cast<int>(i) >= explicit_end.index) {
      break;
    }
    out() << sep;
    FormatArg(inst_id);
  }
  out() << ')';
}

auto Formatter::FormatImportCppDeclRhs() -> void {
  out() << " ";
  OpenBrace();
  for (const Parse::Tree::PackagingNames& import :
       sem_ir_->parse_tree().imports()) {
    if (import.package_id != PackageNameId::Cpp) {
      continue;
    }

    Indent();
    out() << "import Cpp";
    if (import.library_id.has_value()) {
      out() << " \""
            << FormatEscaped(
                   sem_ir_->string_literal_values().Get(import.library_id))
            << "\"";
    } else if (import.inline_body_id.has_value()) {
      out() << " inline";
    }
    out() << "\n";
  }
  CloseBrace();
}

auto Formatter::FormatImportRefRhs(AnyImportRef inst) -> void {
  out() << " ";
  auto import_ir_inst = sem_ir_->import_ir_insts().Get(inst.import_ir_inst_id);
  FormatArg(import_ir_inst.ir_id());
  out() << ", ";
  if (inst.entity_name_id.has_value()) {
    // Prefer to show the entity name when possible.
    FormatArg(inst.entity_name_id);
  } else {
    // Show a name based on the location when possible, or the numeric
    // instruction as a last resort.
    const auto& import_ir = sem_ir_->import_irs().Get(import_ir_inst.ir_id());
    auto loc_id =
        import_ir.sem_ir->insts().GetCanonicalLocId(import_ir_inst.inst_id());
    switch (loc_id.kind()) {
      case LocId::Kind::None: {
        out() << import_ir_inst.inst_id() << " [no loc]";
        break;
      }
      case LocId::Kind::ImportIRInstId: {
        // TODO: Probably don't want to format each indirection, but maybe
        // reuse GetCanonicalImportIRInst?
        out() << import_ir_inst.inst_id() << " [indirect]";
        break;
      }
      case LocId::Kind::NodeId: {
        // Formats a NodeId from the import.
        const auto& tree = import_ir.sem_ir->parse_tree();
        auto token = tree.node_token(loc_id.node_id());
        out() << "loc" << tree.tokens().GetLineNumber(token) << "_"
              << tree.tokens().GetColumnNumber(token);
        break;
      }
      case LocId::Kind::InstId:
        CARBON_FATAL("Unexpected LocId: {0}", loc_id);
    }
  }
  out() << ", "
        << (inst.kind == InstKind::ImportRefLoaded ? "loaded" : "unloaded");
}

auto Formatter::FormatRequireImpls(RequireImplsId id) -> void {
  out() << ' ';

  const auto& require = sem_ir_->require_impls().Get(id);
  OpenBrace();
  Indent();
  out() << "require ";
  FormatArg(require.self_id);
  out() << " impls ";
  FormatArg(require.facet_type_inst_id);
  out() << "\n";
  CloseBrace();
}

auto Formatter::FormatRequireImplsBlock(RequireImplsBlockId block_id) -> void {
  IndentLabel();
  out() << "!requires:\n";
  if (!block_id.has_value()) {
    return;
  }
  for (auto require_impls_id : sem_ir_->require_impls_blocks().Get(block_id)) {
    Indent();
    FormatArg(require_impls_id);
    FormatRequireImpls(require_impls_id);
    out() << "\n";
  }
}

auto Formatter::FormatArg(EntityNameId id) -> void {
  if (!id.has_value()) {
    out() << "_";
    return;
  }
  const auto& info = sem_ir_->entity_names().Get(id);
  FormatName(info.name_id);
  if (info.bind_index().has_value()) {
    out() << ", " << info.bind_index().index;
  }
  if (info.is_template) {
    out() << ", template";
  }
}

auto Formatter::FormatArg(FacetTypeId id) -> void {
  const auto& info = sem_ir_->facet_types().Get(id);
  // Nothing output to indicate that this is a facet type since this is only
  // used as the argument to a `facet_type` instruction.
  out() << "<";

  auto format_specific = [&](SemIR::SpecificId specific_id) {
    if (specific_id.has_value()) {
      out() << ", ";
      FormatName(specific_id);
    }
  };

  llvm::ListSeparator sep(" & ");
  if (info.extend_constraints.empty() &&
      info.extend_named_constraints.empty()) {
    out() << "type";
  } else {
    for (auto extend : info.extend_constraints) {
      out() << sep;
      FormatName(extend.interface_id);
      format_specific(extend.specific_id);
    }
    for (auto extend : info.extend_named_constraints) {
      out() << sep;
      FormatName(extend.named_constraint_id);
      format_specific(extend.specific_id);
    }
  }

  if (info.other_requirements || !info.self_impls_constraints.empty() ||
      !info.type_impls_interfaces.empty() ||
      !info.type_impls_named_constraints.empty() ||
      !info.rewrite_constraints.empty()) {
    out() << " where ";
    llvm::ListSeparator and_sep(" and ");
    int num_self_impls = info.self_impls_constraints.size() +
                         info.self_impls_named_constraints.size();
    if (num_self_impls > 0) {
      out() << and_sep << ".Self impls ";
      llvm::ListSeparator amp_sep(" & ");
      if (num_self_impls > 1) {
        out() << "(";
      }
      for (auto self_impls : info.self_impls_constraints) {
        out() << amp_sep;
        FormatName(self_impls.interface_id);
        format_specific(self_impls.specific_id);
      }
      for (auto self_impls : info.self_impls_named_constraints) {
        out() << amp_sep;
        FormatName(self_impls.named_constraint_id);
        format_specific(self_impls.specific_id);
      }
      if (num_self_impls > 1) {
        out() << ")";
      }
    }
    for (const auto& type_impls : info.type_impls_interfaces) {
      out() << and_sep;
      FormatName(type_impls.self_type);
      out() << " impls ";
      FormatName(type_impls.specific_interface.interface_id);
      format_specific(type_impls.specific_interface.specific_id);
    }
    for (const auto& type_impls : info.type_impls_named_constraints) {
      out() << and_sep;
      FormatName(type_impls.self_type);
      out() << " impls ";
      FormatName(type_impls.specific_named_constraint.named_constraint_id);
      format_specific(type_impls.specific_named_constraint.specific_id);
    }
    for (auto rewrite : info.rewrite_constraints) {
      out() << and_sep;
      FormatArg(rewrite.lhs_id);
      out() << " = ";
      FormatArg(rewrite.rhs_id);
    }
    if (info.other_requirements) {
      out() << and_sep << "TODO";
    }
  }
  out() << ">";
}

auto Formatter::FormatArg(ImportIRId id) -> void {
  if (id.has_value()) {
    out() << GetImportIRLabel(id);
  } else {
    out() << id;
  }
}

auto Formatter::FormatArg(IntId id) -> void {
  // We don't know the signedness to use here. Default to unsigned.
  sem_ir_->ints().Get(id).print(out(), /*isSigned=*/false);
}

auto Formatter::FormatArg(NameScopeId id) -> void {
  OpenBrace();
  FormatNameScope(id);
  CloseBrace();
}

auto Formatter::FormatArg(InstBlockId id) -> void {
  if (!id.has_value()) {
    out() << "invalid";
    return;
  }

  out() << '(';
  llvm::ListSeparator sep;
  for (auto inst_id : sem_ir_->inst_blocks().Get(id)) {
    out() << sep;
    FormatArg(inst_id);
  }
  out() << ')';
}

auto Formatter::FormatArg(AbsoluteInstBlockId id) -> void {
  FormatArg(static_cast<InstBlockId>(id));
}

auto Formatter::FormatArg(ExprRegionId id) -> void {
  const auto& region = sem_ir_->expr_regions().Get(id);

  FormatArg(region.result_id);
  out() << " in ";
  OpenBrace();
  for (auto [i, block_id] : llvm::enumerate(region.block_ids)) {
    if (i != 0) {
      IndentLabel();
      FormatLabel(block_id);
      out() << ":\n";
    }

    FormatCodeBlock(block_id);
  }
  CloseBrace();
}

auto Formatter::FormatArg(RealId id) -> void {
  // TODO: Format with a `.` when the exponent is near zero.
  const auto& real = sem_ir_->reals().Get(id);
  real.mantissa.print(out(), /*isSigned=*/false);
  out() << (real.is_decimal ? 'e' : 'p') << real.exponent;
}

auto Formatter::FormatArg(StringLiteralValueId id) -> void {
  out() << '"'
        << FormatEscaped(sem_ir_->string_literal_values().Get(id),
                         /*use_hex_escapes=*/true)
        << '"';
}

auto Formatter::FormatReturnSlotArg(InstId dest_id) -> void {
  if (dest_id.has_value()) {
    out() << " to ";
    FormatArg(dest_id);
  }
}

auto Formatter::FormatName(NameId id) -> void {
  out() << sem_ir_->names().GetFormatted(id);
}

auto Formatter::FormatName(InstId id) -> void {
  if (id.has_value()) {
    if (auto chunk = tentative_inst_chunks_.Get(id);
        chunk != FormatterChunks::None) {
      chunks_.AppendChildToCurrentParent(chunk);
    }
  }
  out() << inst_namer_.GetNameFor(scope_, id);
}

auto Formatter::FormatName(SpecificId id) -> void {
  const auto& specific = sem_ir_->specifics().Get(id);
  FormatName(specific.generic_id);
  FormatArg(specific.args_id);
}

auto Formatter::FormatName(SpecificInterfaceId id) -> void {
  const auto& interface = sem_ir_->specific_interfaces().Get(id);
  FormatName(interface.interface_id);
  if (interface.specific_id.has_value()) {
    out() << ", ";
    FormatArg(interface.specific_id);
  }
}

auto Formatter::FormatLabel(InstBlockId id) -> void {
  out() << inst_namer_.GetLabelFor(scope_, id);
}

auto Formatter::FormatConstant(ConstantId id) -> void {
  if (!id.has_value()) {
    out() << "<not constant>";
    return;
  }

  auto inst_id = GetInstWithConstantValue(*sem_ir_, id);
  FormatName(inst_id);

  // For an attached constant, also list the unattached constant.
  if (id.is_symbolic() && sem_ir_->constant_values()
                              .GetSymbolicConstant(id)
                              .generic_id.has_value()) {
    // TODO: Skip printing this if it's the same as `inst_id`.
    auto unattached_inst_id = sem_ir_->constant_values().GetInstId(id);
    out() << " (";
    FormatName(unattached_inst_id);
    out() << ")";
  }
}

auto Formatter::FormatInstAsType(InstId id) -> void {
  if (!id.has_value()) {
    out() << "invalid";
    return;
  }

  // Types are formatted in the `constants` scope because they typically refer
  // to constants.
  llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::Constants);
  if (auto const_id = sem_ir_->constant_values().GetAttached(id);
      const_id.has_value()) {
    FormatConstant(const_id);
  } else {
    // Type instruction didn't have a constant value. Fall back to printing
    // the instruction name.
    FormatArg(id);
  }
}

auto Formatter::FormatTypeOfInst(InstId id) -> void {
  auto type_id = sem_ir_->insts().GetAttachedType(id);
  if (!type_id.has_value()) {
    out() << "invalid";
    return;
  }

  // Types are formatted in the `constants` scope because they typically refer
  // to constants.
  llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::Constants);
  FormatConstant(sem_ir_->types().GetConstantId(type_id));
}

auto Formatter::GetImportIRLabel(ImportIRId id) -> std::string {
  CARBON_CHECK(id.has_value(),
               "Callers are responsible for checking `id.has_value`");
  const auto& import_ir = *sem_ir_->import_irs().Get(id).sem_ir;
  CARBON_CHECK(import_ir.library_id().has_value());

  auto package_id = import_ir.package_id();
  llvm::StringRef package_name =
      package_id.AsIdentifierId().has_value()
          ? import_ir.identifiers().Get(package_id.AsIdentifierId())
          : package_id.AsSpecialName();
  llvm::StringRef library_name =
      (import_ir.library_id() != LibraryNameId::Default)
          ? import_ir.string_literal_values().Get(
                import_ir.library_id().AsStringLiteralValueId())
          : "default";
  return llvm::formatv("{0}//{1}", package_name, library_name);
}

}  // namespace Carbon::SemIR

// NOLINTEND(misc-no-recursion)
