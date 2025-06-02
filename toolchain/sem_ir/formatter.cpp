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
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

// TODO: Consider addressing recursion here, although it's not critical because
// the formatter isn't required to work on arbitrary code. Still, it may help
// in the future to debug complex code.
// NOLINTBEGIN(misc-no-recursion)

namespace Carbon::SemIR {

Formatter::Formatter(const File* sem_ir,
                     Parse::GetTreeAndSubtreesFn get_tree_and_subtrees,
                     llvm::ArrayRef<bool> include_ir_in_dumps,
                     bool use_dump_sem_ir_ranges)
    : sem_ir_(sem_ir),
      inst_namer_(sem_ir_),
      get_tree_and_subtrees_(get_tree_and_subtrees),
      include_ir_in_dumps_(include_ir_in_dumps),
      use_dump_sem_ir_ranges_(use_dump_sem_ir_ranges) {
  // Create a placeholder visible chunk and assign it to all instructions that
  // don't have a chunk of their own.
  auto first_chunk = AddChunkNoFlush(true);
  tentative_inst_chunks_.resize(sem_ir_->insts().size(), first_chunk);

  if (use_dump_sem_ir_ranges_) {
    ComputeNodeParents();
  }

  // Create empty placeholder chunks for instructions that we output lazily.
  for (auto inst_id : llvm::concat<const InstId>(
           sem_ir_->constants().array_ref(),
           sem_ir_->inst_blocks().Get(InstBlockId::ImportRefs))) {
    tentative_inst_chunks_[inst_id.index] = AddChunkNoFlush(false);
  }

  // Create a real chunk for the start of the output.
  AddChunkNoFlush(true);
}

auto Formatter::Format() -> void {
  out_ << "--- " << sem_ir_->filename() << "\n";

  FormatTopLevelScopeIfUsed(InstNamer::ScopeId::Constants,
                            sem_ir_->constants().array_ref(),
                            /*use_tentative_output_scopes=*/true);
  FormatTopLevelScopeIfUsed(InstNamer::ScopeId::ImportRefs,
                            sem_ir_->inst_blocks().Get(InstBlockId::ImportRefs),
                            /*use_tentative_output_scopes=*/true);
  FormatTopLevelScopeIfUsed(
      InstNamer::ScopeId::File,
      sem_ir_->inst_blocks().GetOrEmpty(sem_ir_->top_inst_block_id()),
      /*use_tentative_output_scopes=*/false);

  for (auto [id, _] : sem_ir_->interfaces().enumerate()) {
    FormatInterface(id);
  }

  for (auto [id, _] : sem_ir_->associated_constants().enumerate()) {
    FormatAssociatedConstant(id);
  }

  for (auto [id, _] : sem_ir_->impls().enumerate()) {
    FormatImpl(id);
  }

  for (auto [id, _] : sem_ir_->classes().enumerate()) {
    FormatClass(id);
  }

  for (auto [id, _] : sem_ir_->functions().enumerate()) {
    FormatFunction(id);
  }

  for (auto [id, _] : sem_ir_->specifics().enumerate()) {
    FormatSpecific(id);
  }

  out_ << "\n";
}

auto Formatter::ComputeNodeParents() -> void {
  CARBON_CHECK(node_parents_.empty());
  node_parents_.resize(sem_ir_->parse_tree().size(), Parse::NodeId::None);
  for (auto n : sem_ir_->parse_tree().postorder()) {
    for (auto child : get_tree_and_subtrees_().children(n)) {
      node_parents_[child.index] = n;
    }
  }
}

auto Formatter::Write(llvm::raw_ostream& out) -> void {
  FlushChunk();
  for (const auto& chunk : output_chunks_) {
    if (chunk.include_in_output) {
      out << chunk.chunk;
    }
  }
}

auto Formatter::FlushChunk() -> void {
  CARBON_CHECK(output_chunks_.back().chunk.empty());
  output_chunks_.back().chunk = std::move(buffer_);
  buffer_.clear();
}

auto Formatter::AddChunkNoFlush(bool include_in_output) -> size_t {
  CARBON_CHECK(buffer_.empty());
  output_chunks_.push_back({.include_in_output = include_in_output});
  return output_chunks_.size() - 1;
}

auto Formatter::AddChunk(bool include_in_output) -> size_t {
  FlushChunk();
  return AddChunkNoFlush(include_in_output);
}

auto Formatter::IncludeChunkInOutput(size_t chunk) -> void {
  if (chunk == output_chunks_.size() - 1) {
    return;
  }

  if (auto& current_chunk = output_chunks_.back();
      !current_chunk.include_in_output) {
    current_chunk.dependencies.push_back(chunk);
    return;
  }

  llvm::SmallVector<size_t> to_add = {chunk};
  while (!to_add.empty()) {
    auto& chunk = output_chunks_[to_add.pop_back_val()];
    if (chunk.include_in_output) {
      continue;
    }
    chunk.include_in_output = true;
    to_add.append(chunk.dependencies);
    chunk.dependencies.clear();
  }
}

auto Formatter::ShouldIncludeInstByIR(InstId inst_id) -> bool {
  const auto* import_ir = GetCanonicalFileAndInstId(sem_ir_, inst_id).first;
  return include_ir_in_dumps_[import_ir->check_ir_id().index];
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
    end_node_id = node_parents_[end_node_id.index];
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

  out_ << '{';
  indent_ += 2;
  after_open_brace_ = true;
}

auto Formatter::CloseBrace() -> void {
  indent_ -= 2;
  if (!after_open_brace_) {
    Indent();
  }
  out_ << '}';
  after_open_brace_ = false;
}

auto Formatter::Semicolon() -> void {
  FormatPendingImportedFrom(AddSpace::Before);
  out_ << ';';
}

auto Formatter::Indent(int offset) -> void {
  if (after_open_brace_) {
    out_ << '\n';
    after_open_brace_ = false;
  }
  out_.indent(indent_ + offset);
}

auto Formatter::IndentLabel() -> void {
  CARBON_CHECK(indent_ >= 2);
  if (!after_open_brace_) {
    out_ << '\n';
  }
  Indent(-2);
}

auto Formatter::FormatTopLevelScopeIfUsed(InstNamer::ScopeId scope_id,
                                          llvm::ArrayRef<InstId> block,
                                          bool use_tentative_output_scopes)
    -> void {
  if (!use_tentative_output_scopes && use_dump_sem_ir_ranges_) {
    // Don't format the scope if no instructions are in a dump range.
    block = block.drop_while(
        [&](InstId inst_id) { return !ShouldFormatInst(inst_id); });
  }

  if (block.empty()) {
    return;
  }

  llvm::SaveAndRestore scope(scope_, scope_id);
  // Note, we don't use OpenBrace() / CloseBrace() here because we always want
  // a newline to avoid misformatting if the first instruction is omitted.
  out_ << "\n" << inst_namer_.GetScopeName(scope_id) << " {\n";
  indent_ += 2;
  for (const InstId inst_id : block) {
    // Format instructions when needed, but do nothing for elided entries;
    // unlike normal code blocks, scopes are non-sequential so skipped
    // instructions are assumed to be uninteresting.
    if (use_tentative_output_scopes) {
      // This is for constants and imports. These use tentative logic to
      // determine whether an instruction is printed.
      TentativeOutputScope scope(*this, tentative_inst_chunks_[inst_id.index]);
      FormatInst(inst_id);
    } else if (ShouldFormatInst(inst_id)) {
      // This is for the file scope. It uses only the range-based filtering.
      FormatInst(inst_id);
    }
  }
  out_ << "}\n";
  indent_ -= 2;
}

auto Formatter::FormatClass(ClassId id) -> void {
  const Class& class_info = sem_ir_->classes().Get(id);
  if (!ShouldFormatEntity(class_info)) {
    return;
  }

  FormatEntityStart("class", class_info, id);

  llvm::SaveAndRestore class_scope(scope_, inst_namer_.GetScopeFor(id));

  if (class_info.scope_id.has_value()) {
    out_ << ' ';
    OpenBrace();
    FormatCodeBlock(class_info.body_block_id);
    Indent();
    out_ << "complete_type_witness = ";
    FormatName(class_info.complete_type_witness_id);
    out_ << "\n";

    FormatNameScope(class_info.scope_id, "!members:\n");
    CloseBrace();
  } else {
    Semicolon();
  }
  out_ << '\n';

  FormatEntityEnd(class_info.generic_id);
}

auto Formatter::FormatInterface(InterfaceId id) -> void {
  const Interface& interface_info = sem_ir_->interfaces().Get(id);
  if (!ShouldFormatEntity(interface_info)) {
    return;
  }

  FormatEntityStart("interface", interface_info, id);

  llvm::SaveAndRestore interface_scope(scope_, inst_namer_.GetScopeFor(id));

  if (interface_info.scope_id.has_value()) {
    out_ << ' ';
    OpenBrace();
    FormatCodeBlock(interface_info.body_block_id);

    // Always include the !members label because we always list the witness in
    // this section.
    IndentLabel();
    out_ << "!members:\n";
    FormatNameScope(interface_info.scope_id);

    Indent();
    out_ << "witness = ";
    FormatArg(interface_info.associated_entities_id);
    out_ << "\n";

    CloseBrace();
  } else {
    Semicolon();
  }
  out_ << '\n';

  FormatEntityEnd(interface_info.generic_id);
}

auto Formatter::FormatAssociatedConstant(AssociatedConstantId id) -> void {
  const AssociatedConstant& assoc_const =
      sem_ir_->associated_constants().Get(id);
  if (!ShouldFormatEntity(assoc_const.decl_id)) {
    return;
  }

  FormatEntityStart("assoc_const", assoc_const.decl_id, assoc_const.generic_id,
                    id);

  llvm::SaveAndRestore assoc_const_scope(scope_, inst_namer_.GetScopeFor(id));

  out_ << " ";
  FormatName(assoc_const.name_id);
  out_ << ":! ";
  FormatTypeOfInst(assoc_const.decl_id);
  if (assoc_const.default_value_id.has_value()) {
    out_ << " = ";
    FormatArg(assoc_const.default_value_id);
  }
  out_ << ";\n";

  FormatEntityEnd(assoc_const.generic_id);
}

auto Formatter::FormatImpl(ImplId id) -> void {
  const Impl& impl_info = sem_ir_->impls().Get(id);
  if (!ShouldFormatEntity(impl_info)) {
    return;
  }

  FormatEntityStart("impl", impl_info, id);

  llvm::SaveAndRestore impl_scope(scope_, inst_namer_.GetScopeFor(id));

  out_ << ": ";
  FormatName(impl_info.self_id);
  out_ << " as ";
  FormatName(impl_info.constraint_id);

  if (impl_info.is_complete()) {
    out_ << ' ';
    OpenBrace();
    FormatCodeBlock(impl_info.body_block_id);

    // Print the !members label even if the name scope is empty because we
    // always list the witness in this section.
    IndentLabel();
    out_ << "!members:\n";
    if (impl_info.scope_id.has_value()) {
      FormatNameScope(impl_info.scope_id);
    }

    Indent();
    out_ << "witness = ";
    FormatArg(impl_info.witness_id);
    out_ << "\n";

    CloseBrace();
  } else {
    Semicolon();
  }
  out_ << '\n';

  FormatEntityEnd(impl_info.generic_id);
}

auto Formatter::FormatFunction(FunctionId id) -> void {
  const Function& fn = sem_ir_->functions().Get(id);
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
    case FunctionFields::VirtualModifier::Impl:
      function_start += "impl ";
      break;
    case FunctionFields::VirtualModifier::None:
      break;
  }
  if (fn.is_extern) {
    function_start += "extern ";
  }
  function_start += "fn";
  FormatEntityStart(function_start, fn, id);

  llvm::SaveAndRestore function_scope(scope_, inst_namer_.GetScopeFor(id));

  auto return_type_info = ReturnTypeInfo::ForFunction(*sem_ir_, fn);
  FormatParamList(fn.call_params_id, return_type_info.is_valid() &&
                                         return_type_info.has_return_slot());

  if (fn.builtin_function_kind != BuiltinFunctionKind::None) {
    out_ << " = \""
         << FormatEscaped(fn.builtin_function_kind.name(),
                          /*use_hex_escapes=*/true)
         << "\"";
  }

  if (!fn.body_block_ids.empty()) {
    out_ << ' ';
    OpenBrace();

    for (auto block_id : fn.body_block_ids) {
      IndentLabel();
      FormatLabel(block_id);
      out_ << ":\n";

      FormatCodeBlock(block_id);
    }

    CloseBrace();
  } else {
    Semicolon();
  }
  out_ << '\n';

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
    out_ << "!" << region_name << ":\n";
  }
  for (auto [generic_inst_id, specific_inst_id] : llvm::zip_longest(
           sem_ir_->inst_blocks().GetOrEmpty(generic.GetEvalBlock(region)),
           sem_ir_->inst_blocks().GetOrEmpty(specific.GetValueBlock(region)))) {
    Indent();
    if (generic_inst_id) {
      FormatName(*generic_inst_id);
    } else {
      out_ << "<missing>";
    }
    out_ << " => ";
    if (specific_inst_id) {
      FormatName(*specific_inst_id);
    } else {
      out_ << "<missing>";
    }
    out_ << "\n";
  }
}

auto Formatter::FormatSpecific(SpecificId id) -> void {
  const auto& specific = sem_ir_->specifics().Get(id);
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

  out_ << "\n";

  out_ << "specific ";
  FormatName(id);
  out_ << " ";

  OpenBrace();
  FormatSpecificRegion(generic, specific, GenericInstIndex::Region::Declaration,
                       "");
  FormatSpecificRegion(generic, specific, GenericInstIndex::Region::Definition,
                       "definition");
  CloseBrace();

  out_ << "\n";
}

auto Formatter::FormatGenericStart(llvm::StringRef entity_kind,
                                   GenericId generic_id) -> void {
  const auto& generic = sem_ir_->generics().Get(generic_id);
  out_ << "\n";
  Indent();
  out_ << "generic " << entity_kind << " ";
  FormatName(generic_id);

  llvm::SaveAndRestore generic_scope(scope_,
                                     inst_namer_.GetScopeFor(generic_id));

  FormatParamList(generic.bindings_id);

  out_ << " ";
  OpenBrace();
  FormatCodeBlock(generic.decl_block_id);
  if (generic.definition_block_id.has_value()) {
    IndentLabel();
    out_ << "!definition:\n";
    FormatCodeBlock(generic.definition_block_id);
  }
}

auto Formatter::FormatEntityEnd(GenericId generic_id) -> void {
  if (generic_id.has_value()) {
    CloseBrace();
    out_ << '\n';
  }
}

auto Formatter::FormatParamList(InstBlockId params_id, bool has_return_slot)
    -> void {
  if (!params_id.has_value()) {
    // TODO: This happens for imported functions, for which we don't currently
    // import the call parameters list.
    return;
  }

  llvm::StringLiteral close = ")";
  out_ << "(";

  llvm::ListSeparator sep;
  for (InstId param_id : sem_ir_->inst_blocks().Get(params_id)) {
    auto is_out_param = sem_ir_->insts().Is<OutParam>(param_id);
    if (is_out_param) {
      // TODO: An input parameter following an output parameter is formatted a
      // bit strangely. For example, alternating input and output parameters
      // produces:
      //
      //   fn @F(%in1: %t) -> %out1: %t, %in2: %t -> %out2: %t
      //
      // This doesn't actually happen right now, though.
      out_ << std::exchange(close, llvm::StringLiteral(""));
      out_ << " -> ";
    } else {
      out_ << sep;
    }
    if (!param_id.has_value()) {
      out_ << "invalid";
      continue;
    }
    // Don't include the name of the return slot parameter if the function
    // doesn't have a return slot; the name won't be used for anything in that
    // case.
    // TODO: Should the call parameter even exist in that case? There isn't a
    // corresponding argument in a `call` instruction.
    if (!is_out_param || has_return_slot) {
      FormatName(param_id);
      out_ << ": ";
    }
    FormatTypeOfInst(param_id);
  }

  out_ << close;
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
      out_ << "<elided>\n";
      elided = true;
    }
  }
}

auto Formatter::FormatTrailingBlock(InstBlockId block_id) -> void {
  out_ << ' ';
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
    out_ << label;
  }

  for (auto [name_id, result] : scope.entries()) {
    Indent();
    out_ << ".";
    FormatName(name_id);
    switch (result.access_kind()) {
      case AccessKind::Public:
        break;
      case AccessKind::Protected:
        out_ << " [protected]";
        break;
      case AccessKind::Private:
        out_ << " [private]";
        break;
    }
    out_ << " = ";
    if (result.is_poisoned()) {
      out_ << "<poisoned>";
    } else {
      FormatName(result.is_found() ? result.target_inst_id() : InstId::None);
    }
    out_ << "\n";
  }

  for (auto extended_scope_id : scope.extended_scopes()) {
    Indent();
    out_ << "extend ";
    FormatName(extended_scope_id);
    out_ << "\n";
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
    out_ << "import " << label << "\n";
  }

  if (scope.is_cpp_scope()) {
    Indent();
    out_ << "import Cpp//...\n";
  }

  if (scope.has_error()) {
    Indent();
    out_ << "has_error\n";
  }
}

auto Formatter::FormatInst(InstId inst_id) -> void {
  if (!inst_id.has_value()) {
    Indent();
    out_ << "none\n";
    return;
  }

  if (!in_terminator_sequence_) {
    Indent();
  }

  auto inst = sem_ir_->insts().GetWithAttachedType(inst_id);
  CARBON_KIND_SWITCH(inst) {
    case CARBON_KIND(Branch branch): {
      out_ << Branch::Kind.ir_name() << " ";
      FormatLabel(branch.target_id);
      out_ << "\n";
      in_terminator_sequence_ = false;
      return;
    }
    case CARBON_KIND(BranchIf branch_if): {
      out_ << "if ";
      FormatName(branch_if.cond_id);
      out_ << " " << Branch::Kind.ir_name() << " ";
      FormatLabel(branch_if.target_id);
      out_ << " else ";
      in_terminator_sequence_ = true;
      return;
    }
    case CARBON_KIND(BranchWithArg branch_with_arg): {
      out_ << BranchWithArg::Kind.ir_name() << " ";
      FormatLabel(branch_with_arg.target_id);
      out_ << "(";
      FormatName(branch_with_arg.arg_id);
      out_ << ")\n";
      in_terminator_sequence_ = false;
      return;
    }
    default: {
      FormatInstLhs(inst_id, inst);
      out_ << inst.kind().ir_name();

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
      out_ << "\n";
      return;
    }
  }
}

auto Formatter::FormatPendingImportedFrom(AddSpace space_where) -> void {
  if (pending_imported_from_.empty()) {
    return;
  }

  if (space_where == AddSpace::Before) {
    out_ << ' ';
  }
  out_ << "[from \"" << FormatEscaped(pending_imported_from_) << "\"]";
  if (space_where == AddSpace::After) {
    out_ << ' ';
  }
  pending_imported_from_ = llvm::StringRef();
}

auto Formatter::FormatPendingConstantValue(AddSpace space_where) -> void {
  if (pending_constant_value_ == ConstantId::NotConstant) {
    return;
  }

  if (space_where == AddSpace::Before) {
    out_ << ' ';
  }
  out_ << '[';
  if (pending_constant_value_.has_value()) {
    switch (sem_ir_->constant_values().GetDependence(pending_constant_value_)) {
      case ConstantDependence::None:
        out_ << "concrete";
        break;
      case ConstantDependence::PeriodSelf:
        out_ << "symbolic_self";
        break;
      // TODO: Consider renaming this. This will cause a lot of SemIR churn.
      case ConstantDependence::Checked:
        out_ << "symbolic";
        break;
      case ConstantDependence::Template:
        out_ << "template";
        break;
    }
    if (!pending_constant_value_is_self_) {
      out_ << " = ";
      FormatConstant(pending_constant_value_);
    }
  } else {
    out_ << pending_constant_value_;
  }
  out_ << ']';
  if (space_where == AddSpace::After) {
    out_ << ' ';
  }
  pending_constant_value_ = ConstantId::NotConstant;
}

auto Formatter::FormatInstLhs(InstId inst_id, Inst inst) -> void {
  // Every typed instruction is named, and there are some untyped instructions
  // that have names (such as `ImportRefUnloaded`).
  bool has_name = inst_namer_.has_name(inst_id);
  if (!has_name) {
    CARBON_CHECK(!inst.kind().has_type(),
                 "Missing name for typed instruction: {0}", inst);
    return;
  }

  FormatName(inst_id);

  if (inst.kind().has_type()) {
    out_ << ": ";
    switch (GetExprCategory(*sem_ir_, inst_id)) {
      case ExprCategory::NotExpr:
      case ExprCategory::Error:
      case ExprCategory::Value:
      case ExprCategory::Mixed:
        break;
      case ExprCategory::DurableRef:
      case ExprCategory::EphemeralRef:
        out_ << "ref ";
        break;
      case ExprCategory::Initializing:
        out_ << "init ";
        break;
    }
    FormatTypeOfInst(inst_id);
  }

  out_ << " = ";
}

auto Formatter::FormatInstArgAndKind(Inst::ArgAndKind arg_and_kind) -> void {
  GetFormatArgFn(arg_and_kind.kind())(*this, arg_and_kind.value());
}

auto Formatter::FormatInstRhs(Inst inst) -> void {
  CARBON_KIND_SWITCH(inst) {
    case SemIR::InstKind::ArrayInit:
    case SemIR::InstKind::StructInit:
    case SemIR::InstKind::TupleInit: {
      auto init = inst.As<AnyAggregateInit>();
      FormatArgs(init.elements_id);
      FormatReturnSlotArg(init.dest_id);
      return;
    }

    case SemIR::InstKind::ImportRefLoaded:
    case SemIR::InstKind::ImportRefUnloaded:
      FormatImportRefRhs(inst.As<AnyImportRef>());
      return;

    case SemIR::InstKind::OutParam:
    case SemIR::InstKind::RefParam:
    case SemIR::InstKind::ValueParam: {
      auto param = inst.As<AnyParam>();
      FormatArgs(param.index);
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

    case CARBON_KIND(BindSymbolicName bind): {
      // A BindSymbolicName with no value is a purely symbolic binding, such as
      // the `Self` in an interface. Don't print out `none` for the value.
      if (bind.value_id.has_value()) {
        FormatArgs(bind.entity_name_id, bind.value_id);
      } else {
        FormatArgs(bind.entity_name_id);
      }
      return;
    }

    case CARBON_KIND(BlockArg block): {
      out_ << " ";
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

    case CARBON_KIND(FloatLiteral value): {
      llvm::SmallVector<char, 16> buffer;
      sem_ir_->floats().Get(value.float_id).toString(buffer);
      out_ << " " << buffer;
      return;
    }

    case CARBON_KIND(FunctionDecl decl): {
      FormatDeclRhs(decl.function_id,
                    sem_ir_->functions().Get(decl.function_id).pattern_block_id,
                    decl.decl_block_id);
      return;
    }

    case InstKind::ImportCppDecl: {
      FormatImportCppDeclRhs();
      return;
    }

    case CARBON_KIND(ImplDecl decl): {
      FormatDeclRhs(decl.impl_id,
                    sem_ir_->impls().Get(decl.impl_id).pattern_block_id,
                    decl.decl_block_id);
      return;
    }

    case CARBON_KIND(InitializeFrom init): {
      FormatArgs(init.src_id);
      FormatReturnSlotArg(init.dest_id);
      return;
    }

    case CARBON_KIND(InstValue inst): {
      out_ << ' ';
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
      out_ << " ";
      sem_ir_->ints()
          .Get(value.int_id)
          .print(out_, sem_ir_->types().IsSignedInt(value.type_id));
      return;
    }

    case CARBON_KIND(NameBindingDecl name): {
      FormatTrailingBlock(name.pattern_block_id);
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

    case InstKind::ReturnSlotPattern:
      // No-op because type_id is the only semantically significant field,
      // and it's handled separately.
      return;

    case CARBON_KIND(SpliceBlock splice): {
      FormatArgs(splice.result_id);
      FormatTrailingBlock(splice.block_id);
      return;
    }

    case CARBON_KIND(StructType struct_type): {
      out_ << " {";
      llvm::ListSeparator sep;
      for (auto field :
           sem_ir_->struct_type_fields().Get(struct_type.fields_id)) {
        out_ << sep << ".";
        FormatName(field.name_id);
        out_ << ": ";
        FormatInstAsType(field.type_inst_id);
      }
      out_ << "}";
      return;
    }

    case CARBON_KIND(WhereExpr where): {
      FormatArgs(where.period_self_id);
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
  out_ << " ";
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
  out_ << ", ";
  FormatInstArgAndKind(arg1);
}

auto Formatter::FormatCallRhs(Call inst) -> void {
  out_ << " ";
  FormatArg(inst.callee_id);

  if (!inst.args_id.has_value()) {
    out_ << "(<none>)";
    return;
  }

  llvm::ArrayRef<InstId> args = sem_ir_->inst_blocks().Get(inst.args_id);

  auto return_info = ReturnTypeInfo::ForType(*sem_ir_, inst.type_id);
  if (!return_info.is_valid()) {
    out_ << "(<invalid return info>)";
    return;
  }
  bool has_return_slot = return_info.has_return_slot();
  InstId return_slot_arg_id = InstId::None;
  if (has_return_slot) {
    return_slot_arg_id = args.back();
    args = args.drop_back();
  }

  llvm::ListSeparator sep;
  out_ << '(';
  for (auto inst_id : args) {
    out_ << sep;
    FormatArg(inst_id);
  }
  out_ << ')';

  if (has_return_slot) {
    FormatReturnSlotArg(return_slot_arg_id);
  }
}

auto Formatter::FormatImportCppDeclRhs() -> void {
  out_ << " ";
  OpenBrace();
  for (ImportCpp import_cpp : sem_ir_->import_cpps().values()) {
    Indent();
    out_ << "import Cpp \""
         << FormatEscaped(
                sem_ir_->string_literal_values().Get(import_cpp.library_id))
         << "\"\n";
  }
  CloseBrace();
}

auto Formatter::FormatImportRefRhs(AnyImportRef inst) -> void {
  out_ << " ";
  auto import_ir_inst = sem_ir_->import_ir_insts().Get(inst.import_ir_inst_id);
  FormatArg(import_ir_inst.ir_id());
  out_ << ", ";
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
        out_ << import_ir_inst.inst_id() << " [no loc]";
        break;
      }
      case LocId::Kind::ImportIRInstId: {
        // TODO: Probably don't want to format each indirection, but maybe
        // reuse GetCanonicalImportIRInst?
        out_ << import_ir_inst.inst_id() << " [indirect]";
        break;
      }
      case LocId::Kind::NodeId: {
        // Formats a NodeId from the import.
        const auto& tree = import_ir.sem_ir->parse_tree();
        auto token = tree.node_token(loc_id.node_id());
        out_ << "loc" << tree.tokens().GetLineNumber(token) << "_"
             << tree.tokens().GetColumnNumber(token);
        break;
      }
      case LocId::Kind::InstId:
        CARBON_FATAL("Unexpected LocId: {0}", loc_id);
    }
  }
  out_ << ", "
       << (inst.kind == InstKind::ImportRefLoaded ? "loaded" : "unloaded");
}

auto Formatter::FormatArg(EntityNameId id) -> void {
  if (!id.has_value()) {
    out_ << "_";
    return;
  }
  const auto& info = sem_ir_->entity_names().Get(id);
  FormatName(info.name_id);
  if (info.bind_index().has_value()) {
    out_ << ", " << info.bind_index().index;
  }
  if (info.is_template) {
    out_ << ", template";
  }
}

auto Formatter::FormatArg(FacetTypeId id) -> void {
  const auto& info = sem_ir_->facet_types().Get(id);
  // Nothing output to indicate that this is a facet type since this is only
  // used as the argument to a `facet_type` instruction.
  out_ << "<";

  llvm::ListSeparator sep(" & ");
  if (info.extend_constraints.empty()) {
    out_ << "type";
  } else {
    for (auto interface : info.extend_constraints) {
      out_ << sep;
      FormatName(interface.interface_id);
      if (interface.specific_id.has_value()) {
        out_ << ", ";
        FormatName(interface.specific_id);
      }
    }
  }

  if (info.other_requirements || !info.self_impls_constraints.empty() ||
      !info.rewrite_constraints.empty()) {
    out_ << " where ";
    llvm::ListSeparator and_sep(" and ");
    if (!info.self_impls_constraints.empty()) {
      out_ << and_sep << ".Self impls ";
      llvm::ListSeparator amp_sep(" & ");
      for (auto interface : info.self_impls_constraints) {
        out_ << amp_sep;
        FormatName(interface.interface_id);
        if (interface.specific_id.has_value()) {
          out_ << ", ";
          FormatName(interface.specific_id);
        }
      }
    }
    for (auto rewrite : info.rewrite_constraints) {
      out_ << and_sep;
      FormatArg(rewrite.lhs_id);
      out_ << " = ";
      FormatArg(rewrite.rhs_id);
    }
    if (info.other_requirements) {
      out_ << and_sep << "TODO";
    }
  }
  out_ << ">";
}

auto Formatter::FormatArg(ImportIRId id) -> void {
  if (id.has_value()) {
    out_ << GetImportIRLabel(id);
  } else {
    out_ << id;
  }
}

auto Formatter::FormatArg(IntId id) -> void {
  // We don't know the signedness to use here. Default to unsigned.
  sem_ir_->ints().Get(id).print(out_, /*isSigned=*/false);
}

auto Formatter::FormatArg(NameScopeId id) -> void {
  OpenBrace();
  FormatNameScope(id);
  CloseBrace();
}

auto Formatter::FormatArg(InstBlockId id) -> void {
  if (!id.has_value()) {
    out_ << "invalid";
    return;
  }

  out_ << '(';
  llvm::ListSeparator sep;
  for (auto inst_id : sem_ir_->inst_blocks().Get(id)) {
    out_ << sep;
    FormatArg(inst_id);
  }
  out_ << ')';
}

auto Formatter::FormatArg(AbsoluteInstBlockId id) -> void {
  FormatArg(static_cast<InstBlockId>(id));
}

auto Formatter::FormatArg(RealId id) -> void {
  // TODO: Format with a `.` when the exponent is near zero.
  const auto& real = sem_ir_->reals().Get(id);
  real.mantissa.print(out_, /*isSigned=*/false);
  out_ << (real.is_decimal ? 'e' : 'p') << real.exponent;
}

auto Formatter::FormatArg(StringLiteralValueId id) -> void {
  out_ << '"'
       << FormatEscaped(sem_ir_->string_literal_values().Get(id),
                        /*use_hex_escapes=*/true)
       << '"';
}

auto Formatter::FormatReturnSlotArg(InstId dest_id) -> void {
  out_ << " to ";
  FormatArg(dest_id);
}

auto Formatter::FormatName(NameId id) -> void {
  out_ << sem_ir_->names().GetFormatted(id);
}

auto Formatter::FormatName(InstId id) -> void {
  if (id.has_value()) {
    IncludeChunkInOutput(tentative_inst_chunks_[id.index]);
  }
  out_ << inst_namer_.GetNameFor(scope_, id);
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
    out_ << ", ";
    FormatArg(interface.specific_id);
  }
}

auto Formatter::FormatLabel(InstBlockId id) -> void {
  out_ << inst_namer_.GetLabelFor(scope_, id);
}

auto Formatter::FormatConstant(ConstantId id) -> void {
  if (!id.has_value()) {
    out_ << "<not constant>";
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
    out_ << " (";
    FormatName(unattached_inst_id);
    out_ << ")";
  }
}

auto Formatter::FormatInstAsType(InstId id) -> void {
  if (!id.has_value()) {
    out_ << "invalid";
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
    out_ << "invalid";
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
