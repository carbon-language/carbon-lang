// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter.h"

#include "common/ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_namer.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

// TODO: Consider addressing recursion here, although it's not critical because
// the formatter isn't required to work on arbitrary code. Still, it may help
// in the future to debug complex code.
// NOLINTBEGIN(misc-no-recursion)

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class FormatterImpl {
 public:
  explicit FormatterImpl(const File* sem_ir, InstNamer* inst_namer,
                         Formatter::ShouldFormatEntityFn should_format_entity,
                         int indent)
      : sem_ir_(sem_ir),
        inst_namer_(inst_namer),
        should_format_entity_(should_format_entity),
        indent_(indent) {
    // Create the first chunk and assign it to all instructions that don't have
    // a chunk of their own.
    auto first_chunk = AddChunkNoFlush(true);
    tentative_inst_chunks_.resize(sem_ir_->insts().size(), first_chunk);
  }

  // Prints the SemIR.
  //
  // Constants are printed first and may be referenced by later sections,
  // including file-scoped instructions. The file scope may contain entity
  // declarations which are defined later, such as classes.
  auto Format() -> void {
    out_ << "--- " << sem_ir_->filename() << "\n\n";

    FormatScopeIfUsed(InstNamer::ScopeId::Constants,
                      sem_ir_->constants().array_ref());
    FormatScopeIfUsed(InstNamer::ScopeId::ImportRefs,
                      sem_ir_->inst_blocks().Get(InstBlockId::ImportRefs));

    out_ << inst_namer_->GetScopeName(InstNamer::ScopeId::File) << " ";
    OpenBrace();

    // TODO: Handle the case where there are multiple top-level instruction
    // blocks. For example, there may be branching in the initializer of a
    // global or a type expression.
    if (auto block_id = sem_ir_->top_inst_block_id(); block_id.has_value()) {
      llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::File);
      FormatCodeBlock(block_id);
    }

    CloseBrace();
    out_ << '\n';

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

    // End-of-file newline.
    out_ << "\n";
  }

  // Write buffered output to the given stream.
  auto Write(llvm::raw_ostream& out) -> void {
    FlushChunk();
    for (const auto& chunk : output_chunks_) {
      if (chunk.include_in_output) {
        out << chunk.chunk;
      }
    }
  }

 private:
  enum class AddSpace : bool { Before, After };

  // A chunk of the buffered output. Chunks of the output, such as constant
  // values, are buffered until we reach the end of formatting so that we can
  // decide whether to include them based on whether they are referenced.
  struct OutputChunk {
    // Whether this chunk is known to be included in the output.
    bool include_in_output;
    // The textual contents of this chunk.
    std::string chunk = std::string();
    // Chunks that should be included in the output if this one is.
    llvm::SmallVector<size_t> dependencies = {};
  };

  // A scope in which output should be buffered because we don't yet know
  // whether to include it in the final formatted SemIR.
  struct TentativeOutputScope {
    explicit TentativeOutputScope(FormatterImpl& f) : formatter(f) {
      index = formatter.AddChunk(false);
    }
    ~TentativeOutputScope() {
      auto next_index = formatter.AddChunk(true);
      CARBON_CHECK(next_index == index + 1, "Nested TentativeOutputScope");
    }
    FormatterImpl& formatter;
    size_t index;
  };

  // Flushes the buffered output to the current chunk.
  auto FlushChunk() -> void {
    CARBON_CHECK(output_chunks_.back().chunk.empty());
    output_chunks_.back().chunk = std::move(buffer_);
    buffer_.clear();
  }

  // Adds a new chunk to the output. Does not flush existing output, so should
  // only be called if there is no buffered output.
  auto AddChunkNoFlush(bool include_in_output) -> size_t {
    CARBON_CHECK(buffer_.empty());
    output_chunks_.push_back({.include_in_output = include_in_output});
    return output_chunks_.size() - 1;
  }

  // Flushes the current chunk and add a new chunk to the output.
  auto AddChunk(bool include_in_output) -> size_t {
    FlushChunk();
    return AddChunkNoFlush(include_in_output);
  }

  // Marks the given chunk as being included in the output if the current chunk
  // is.
  auto IncludeChunkInOutput(size_t chunk) -> void {
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

  // Determines whether the specified entity should be included in the formatted
  // output.
  auto ShouldFormatEntity(SemIR::InstId decl_id) -> bool {
    if (!decl_id.has_value()) {
      return true;
    }
    return should_format_entity_(decl_id);
  }

  auto ShouldFormatEntity(const EntityWithParamsBase& entity) -> bool {
    return ShouldFormatEntity(entity.latest_decl_id());
  }

  // Begins a braced block. Writes an open brace, and prepares to insert a
  // newline after it if the braced block is non-empty.
  auto OpenBrace() -> void {
    // Put the constant value of an instruction before any braced block, rather
    // than at the end.
    FormatPendingConstantValue(AddSpace::After);

    // Put the imported-from library name before the definition of the entity.
    FormatPendingImportedFrom(AddSpace::After);

    out_ << '{';
    indent_ += 2;
    after_open_brace_ = true;
  }

  // Ends a braced block by writing a close brace.
  auto CloseBrace() -> void {
    indent_ -= 2;
    if (!after_open_brace_) {
      Indent();
    }
    out_ << '}';
    after_open_brace_ = false;
  }

  auto Semicolon() -> void {
    FormatPendingImportedFrom(AddSpace::Before);
    out_ << ';';
  }

  // Adds beginning-of-line indentation. If we're at the start of a braced
  // block, first starts a new line.
  auto Indent(int offset = 0) -> void {
    if (after_open_brace_) {
      out_ << '\n';
      after_open_brace_ = false;
    }
    out_.indent(indent_ + offset);
  }

  // Adds beginning-of-label indentation. This is one level less than normal
  // indentation. Labels also get a preceding blank line unless they're at the
  // start of a block.
  auto IndentLabel() -> void {
    CARBON_CHECK(indent_ >= 2);
    if (!after_open_brace_) {
      out_ << '\n';
    }
    Indent(-2);
  }

  // Formats a top-level scope, and any of the instructions in that scope that
  // are used.
  auto FormatScopeIfUsed(InstNamer::ScopeId scope_id,
                         llvm::ArrayRef<InstId> block) -> void {
    if (block.empty()) {
      return;
    }

    llvm::SaveAndRestore scope(scope_, scope_id);
    // Note, we don't use OpenBrace() / CloseBrace() here because we always want
    // a newline to avoid misformatting if the first instruction is omitted.
    out_ << inst_namer_->GetScopeName(scope_id) << " {\n";
    indent_ += 2;
    for (const InstId inst_id : block) {
      TentativeOutputScope scope(*this);
      tentative_inst_chunks_[inst_id.index] = scope.index;
      FormatInst(inst_id);
    }
    out_ << "}\n\n";
    indent_ -= 2;
  }

  // Formats a full class.
  auto FormatClass(ClassId id) -> void {
    const Class& class_info = sem_ir_->classes().Get(id);
    if (!ShouldFormatEntity(class_info)) {
      return;
    }

    FormatEntityStart("class", class_info, id);

    llvm::SaveAndRestore class_scope(scope_, inst_namer_->GetScopeFor(id));

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

  // Formats a full interface.
  auto FormatInterface(InterfaceId id) -> void {
    const Interface& interface_info = sem_ir_->interfaces().Get(id);
    if (!ShouldFormatEntity(interface_info)) {
      return;
    }

    FormatEntityStart("interface", interface_info, id);

    llvm::SaveAndRestore interface_scope(scope_, inst_namer_->GetScopeFor(id));

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

  // Formats an associated constant entity.
  auto FormatAssociatedConstant(AssociatedConstantId id) -> void {
    const AssociatedConstant& assoc_const =
        sem_ir_->associated_constants().Get(id);
    if (!ShouldFormatEntity(assoc_const.decl_id)) {
      return;
    }

    FormatEntityStart("assoc_const", assoc_const.decl_id,
                      assoc_const.generic_id, id);

    llvm::SaveAndRestore assoc_const_scope(scope_,
                                           inst_namer_->GetScopeFor(id));

    out_ << " ";
    FormatName(assoc_const.name_id);
    out_ << ":! ";
    FormatArg(sem_ir_->insts().Get(assoc_const.decl_id).type_id());
    if (assoc_const.default_value_id.has_value()) {
      out_ << " = ";
      FormatArg(assoc_const.default_value_id);
    }
    out_ << ";\n";

    FormatEntityEnd(assoc_const.generic_id);
  }

  // Formats a full impl.
  auto FormatImpl(ImplId id) -> void {
    const Impl& impl_info = sem_ir_->impls().Get(id);
    if (!ShouldFormatEntity(impl_info)) {
      return;
    }

    FormatEntityStart("impl", impl_info, id);

    llvm::SaveAndRestore impl_scope(scope_, inst_namer_->GetScopeFor(id));

    out_ << ": ";
    FormatName(impl_info.self_id);
    out_ << " as ";
    FormatName(impl_info.constraint_id);

    if (impl_info.is_defined()) {
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

  // Formats a full function.
  auto FormatFunction(FunctionId id) -> void {
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

    llvm::SaveAndRestore function_scope(scope_, inst_namer_->GetScopeFor(id));

    FormatParamList(fn.implicit_param_patterns_id, /*is_implicit=*/true);
    FormatParamList(fn.param_patterns_id, /*is_implicit=*/false);

    if (fn.return_slot_pattern_id.has_value()) {
      out_ << " -> ";
      auto return_info = ReturnTypeInfo::ForFunction(*sem_ir_, fn);
      if (!fn.body_block_ids.empty() && return_info.is_valid() &&
          return_info.has_return_slot()) {
        FormatName(fn.return_slot_pattern_id);
        out_ << ": ";
      }
      FormatType(sem_ir_->insts().Get(fn.return_slot_pattern_id).type_id());
    }

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

  // Helper for FormatSpecific to print regions.
  auto FormatSpecificRegion(const Generic& generic, const Specific& specific,
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
             sem_ir_->inst_blocks().GetOrEmpty(
                 specific.GetValueBlock(region)))) {
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

  // Formats a full specific.
  auto FormatSpecific(SpecificId id) -> void {
    const auto& specific = sem_ir_->specifics().Get(id);
    const auto& generic = sem_ir_->generics().Get(specific.generic_id);
    if (!should_format_entity_(generic.decl_id)) {
      // Omit specifics if we also omitted the generic.
      return;
    }

    llvm::SaveAndRestore generic_scope(
        scope_, inst_namer_->GetScopeFor(specific.generic_id));

    out_ << "\n";

    out_ << "specific ";
    FormatName(id);
    out_ << " ";

    OpenBrace();
    FormatSpecificRegion(generic, specific,
                         GenericInstIndex::Region::Declaration, "");
    FormatSpecificRegion(generic, specific,
                         GenericInstIndex::Region::Definition, "definition");
    CloseBrace();

    out_ << "\n";
  }

  // Handles generic-specific setup for FormatEntityStart.
  auto FormatGenericStart(llvm::StringRef entity_kind, GenericId generic_id)
      -> void {
    const auto& generic = sem_ir_->generics().Get(generic_id);
    out_ << "\n";
    Indent();
    out_ << "generic " << entity_kind << " ";
    FormatName(generic_id);

    llvm::SaveAndRestore generic_scope(scope_,
                                       inst_namer_->GetScopeFor(generic_id));

    FormatParamList(generic.bindings_id, /*is_implicit=*/false);

    out_ << " ";
    OpenBrace();
    FormatCodeBlock(generic.decl_block_id);
    if (generic.definition_block_id.has_value()) {
      IndentLabel();
      out_ << "!definition:\n";
      FormatCodeBlock(generic.definition_block_id);
    }
  }

  // Provides common formatting for entities, paired with FormatEntityEnd.
  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind,
                         InstId first_owning_decl_id, GenericId generic_id,
                         IdT entity_id) -> void {
    // If this entity was imported from a different IR, annotate the name of
    // that IR in the output before the `{` or `;`.
    if (first_owning_decl_id.has_value()) {
      auto loc_id = sem_ir_->insts().GetLocId(first_owning_decl_id);
      if (loc_id.is_import_ir_inst_id()) {
        auto import_ir_id =
            sem_ir_->import_ir_insts().Get(loc_id.import_ir_inst_id()).ir_id;
        const auto* import_file =
            sem_ir_->import_irs().Get(import_ir_id).sem_ir;
        pending_imported_from_ = import_file->filename();
      }
    }

    if (generic_id.has_value()) {
      FormatGenericStart(entity_kind, generic_id);
    }

    out_ << "\n";
    after_open_brace_ = false;
    Indent();
    out_ << entity_kind;

    // If there's a generic, it will have attached the name. Otherwise, add the
    // name here.
    if (!generic_id.has_value()) {
      out_ << " ";
      FormatName(entity_id);
    }
  }

  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind,
                         const EntityWithParamsBase& entity, IdT entity_id)
      -> void {
    FormatEntityStart(entity_kind, entity.first_owning_decl_id,
                      entity.generic_id, entity_id);
  }

  // Provides common formatting for entities, paired with FormatEntityStart.
  auto FormatEntityEnd(GenericId generic_id) -> void {
    if (generic_id.has_value()) {
      CloseBrace();
      out_ << '\n';
    }
  }

  // Formats parameters, eliding them completely if they're empty. Wraps in
  // parentheses or square brackets based on whether these are implicit
  // parameters.
  auto FormatParamList(InstBlockId param_patterns_id, bool is_implicit)
      -> void {
    if (!param_patterns_id.has_value()) {
      return;
    }

    out_ << (is_implicit ? "[" : "(");

    llvm::ListSeparator sep;
    for (InstId param_id : sem_ir_->inst_blocks().Get(param_patterns_id)) {
      out_ << sep;
      if (!param_id.has_value()) {
        out_ << "invalid";
        continue;
      }
      if (auto addr = sem_ir_->insts().TryGetAs<SemIR::AddrPattern>(param_id)) {
        out_ << "addr ";
        param_id = addr->inner_id;
      }
      FormatName(param_id);
      out_ << ": ";
      FormatType(sem_ir_->insts().Get(param_id).type_id());
    }

    out_ << (is_implicit ? "]" : ")");
  }

  // Prints instructions for a code block.
  auto FormatCodeBlock(InstBlockId block_id) -> void {
    for (const InstId inst_id : sem_ir_->inst_blocks().GetOrEmpty(block_id)) {
      FormatInst(inst_id);
    }
  }

  // Prints a code block with braces, intended to be used trailing after other
  // content on the same line. If non-empty, instructions are on separate lines.
  auto FormatTrailingBlock(InstBlockId block_id) -> void {
    out_ << ' ';
    OpenBrace();
    FormatCodeBlock(block_id);
    CloseBrace();
  }

  // Prints the contents of a name scope, with an optional label.
  auto FormatNameScope(NameScopeId id, llvm::StringRef label = "") -> void {
    const auto& scope = sem_ir_->name_scopes().Get(id);

    if (scope.entries().empty() && scope.extended_scopes().empty() &&
        scope.import_ir_scopes().empty() && !scope.has_error()) {
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
        case SemIR::AccessKind::Public:
          break;
        case SemIR::AccessKind::Protected:
          out_ << " [protected]";
          break;
        case SemIR::AccessKind::Private:
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

    if (scope.has_error()) {
      Indent();
      out_ << "has_error\n";
    }
  }

  // Prints a single instruction.
  auto FormatInst(InstId inst_id) -> void {
    if (!inst_id.has_value()) {
      Indent();
      out_ << "none\n";
      return;
    }

    FormatInst(inst_id, sem_ir_->insts().Get(inst_id));
  }

  auto FormatInst(InstId inst_id, Inst inst) -> void {
    CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(InstT)  \
  case CARBON_KIND(InstT typed_inst): { \
    FormatInst(inst_id, typed_inst);    \
    break;                              \
  }
#include "toolchain/sem_ir/inst_kind.def"
    }
  }

  template <typename InstT>
  auto FormatInst(InstId inst_id, InstT inst) -> void {
    Indent();
    FormatInstLhs(inst_id, inst);
    out_ << InstT::Kind.ir_name();
    pending_constant_value_ = sem_ir_->constant_values().Get(inst_id);
    pending_constant_value_is_self_ =
        sem_ir_->constant_values().GetInstIdIfValid(pending_constant_value_) ==
        inst_id;
    FormatInstRhs(inst);
    FormatPendingConstantValue(AddSpace::Before);
    out_ << "\n";
  }

  // Don't print a constant for ImportRefUnloaded.
  auto FormatInst(InstId inst_id, ImportRefUnloaded inst) -> void {
    Indent();
    FormatInstLhs(inst_id, inst);
    out_ << ImportRefUnloaded::Kind.ir_name();
    FormatInstRhs(inst);
    out_ << "\n";
  }

  // If there is a pending library name that the current instruction was
  // imported from, print it now and clear it out.
  auto FormatPendingImportedFrom(AddSpace space_where) -> void {
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

  // If there is a pending constant value attached to the current instruction,
  // print it now and clear it out. The constant value gets printed before the
  // first braced block argument, or at the end of the instruction if there are
  // no such arguments.
  auto FormatPendingConstantValue(AddSpace space_where) -> void {
    if (pending_constant_value_ == ConstantId::NotConstant) {
      return;
    }

    if (space_where == AddSpace::Before) {
      out_ << ' ';
    }
    out_ << '[';
    if (pending_constant_value_.has_value()) {
      switch (
          sem_ir_->constant_values().GetDependence(pending_constant_value_)) {
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

  auto FormatInstLhs(InstId inst_id, Inst inst) -> void {
    switch (inst.kind().value_kind()) {
      case InstValueKind::Typed:
        FormatName(inst_id);
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
        FormatType(inst.type_id());
        out_ << " = ";
        break;
      case InstValueKind::None:
        break;
    }
  }

  // Format ImportCppDecl name.
  auto FormatInstLhs(InstId inst_id, ImportCppDecl /*inst*/) -> void {
    FormatName(inst_id);
    out_ << " = ";
  }

  // Format ImportDecl with its name.
  auto FormatInstLhs(InstId inst_id, ImportDecl /*inst*/) -> void {
    FormatName(inst_id);
    out_ << " = ";
  }

  // Print ImportRefUnloaded with type-like semantics even though it lacks a
  // type_id.
  auto FormatInstLhs(InstId inst_id, ImportRefUnloaded /*inst*/) -> void {
    FormatName(inst_id);
    out_ << " = ";
  }

  template <typename InstT>
  auto FormatInstRhs(InstT inst) -> void {
    // By default, an instruction has a comma-separated argument list.
    using Info = Internal::InstLikeTypeInfo<InstT>;
    if constexpr (Info::NumArgs == 2) {
      // Several instructions have a second operand that's a specific ID. We
      // don't include it in the argument list if there is no corresponding
      // specific, that is, when we're not in a generic context.
      if constexpr (std::is_same_v<typename Info::template ArgType<1>,
                                   SemIR::SpecificId>) {
        if (!Info::template Get<1>(inst).has_value()) {
          FormatArgs(Info::template Get<0>(inst));
          return;
        }
      }
      FormatArgs(Info::template Get<0>(inst), Info::template Get<1>(inst));
    } else if constexpr (Info::NumArgs == 1) {
      FormatArgs(Info::template Get<0>(inst));
    } else {
      FormatArgs();
    }
  }

  auto FormatInstRhs(BindSymbolicName inst) -> void {
    // A BindSymbolicName with no value is a purely symbolic binding, such as
    // the `Self` in an interface. Don't print out `none` for the value.
    if (inst.value_id.has_value()) {
      FormatArgs(inst.entity_name_id, inst.value_id);
    } else {
      FormatArgs(inst.entity_name_id);
    }
  }

  auto FormatInstRhs(BlockArg inst) -> void {
    out_ << " ";
    FormatLabel(inst.block_id);
  }

  auto FormatInstRhs(Namespace inst) -> void {
    if (inst.import_id.has_value()) {
      FormatArgs(inst.import_id, inst.name_scope_id);
    } else {
      FormatArgs(inst.name_scope_id);
    }
  }

  auto FormatInst(InstId /*inst_id*/, BranchIf inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << "if ";
    FormatName(inst.cond_id);
    out_ << " " << Branch::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << " else ";
    in_terminator_sequence_ = true;
  }

  auto FormatInst(InstId /*inst_id*/, BranchWithArg inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << BranchWithArg::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << "(";
    FormatName(inst.arg_id);
    out_ << ")\n";
    in_terminator_sequence_ = false;
  }

  auto FormatInst(InstId /*inst_id*/, Branch inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << Branch::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << "\n";
    in_terminator_sequence_ = false;
  }

  auto FormatInstRhs(Call inst) -> void {
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

  auto FormatInstRhs(ArrayInit inst) -> void {
    FormatArgs(inst.inits_id);
    FormatReturnSlotArg(inst.dest_id);
  }

  auto FormatInstRhs(InitializeFrom inst) -> void {
    FormatArgs(inst.src_id);
    FormatReturnSlotArg(inst.dest_id);
  }

  auto FormatInstRhs(ValueParam inst) -> void {
    FormatArgs(inst.index);
    // Omit pretty_name because it's an implementation detail of
    // pretty-printing.
  }

  auto FormatInstRhs(RefParam inst) -> void {
    FormatArgs(inst.index);
    // Omit pretty_name because it's an implementation detail of
    // pretty-printing.
  }

  auto FormatInstRhs(OutParam inst) -> void {
    FormatArgs(inst.index);
    // Omit pretty_name because it's an implementation detail of
    // pretty-printing.
  }

  auto FormatInstRhs(ReturnExpr ret) -> void {
    FormatArgs(ret.expr_id);
    if (ret.dest_id.has_value()) {
      FormatReturnSlotArg(ret.dest_id);
    }
  }

  auto FormatInstRhs(ReturnSlot inst) -> void {
    // Omit inst.type_inst_id because it's not semantically significant.
    FormatArgs(inst.storage_id);
  }

  auto FormatInstRhs(ReturnSlotPattern /*inst*/) -> void {
    // No-op because type_id is the only semantically significant field,
    // and it's handled separately.
  }

  auto FormatInstRhs(StructInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlotArg(init.dest_id);
  }

  auto FormatInstRhs(TupleInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlotArg(init.dest_id);
  }

  auto FormatInstRhs(FunctionDecl inst) -> void {
    FormatArgs(inst.function_id);
    llvm::SaveAndRestore class_scope(
        scope_, inst_namer_->GetScopeFor(inst.function_id));
    FormatTrailingBlock(
        sem_ir_->functions().Get(inst.function_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRhs(ClassDecl inst) -> void {
    FormatArgs(inst.class_id);
    llvm::SaveAndRestore class_scope(scope_,
                                     inst_namer_->GetScopeFor(inst.class_id));
    FormatTrailingBlock(sem_ir_->classes().Get(inst.class_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRhs(ImplDecl inst) -> void {
    FormatArgs(inst.impl_id);
    llvm::SaveAndRestore class_scope(scope_,
                                     inst_namer_->GetScopeFor(inst.impl_id));
    FormatTrailingBlock(sem_ir_->impls().Get(inst.impl_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRhs(InterfaceDecl inst) -> void {
    FormatArgs(inst.interface_id);
    llvm::SaveAndRestore class_scope(
        scope_, inst_namer_->GetScopeFor(inst.interface_id));
    FormatTrailingBlock(
        sem_ir_->interfaces().Get(inst.interface_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRhs(AssociatedConstantDecl inst) -> void {
    FormatArgs(inst.assoc_const_id);
    llvm::SaveAndRestore assoc_const_scope(
        scope_, inst_namer_->GetScopeFor(inst.assoc_const_id));
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRhs(IntValue inst) -> void {
    out_ << " ";
    sem_ir_->ints()
        .Get(inst.int_id)
        .print(out_, sem_ir_->types().IsSignedInt(inst.type_id));
  }

  auto FormatInstRhs(FloatLiteral inst) -> void {
    llvm::SmallVector<char, 16> buffer;
    sem_ir_->floats().Get(inst.float_id).toString(buffer);
    out_ << " " << buffer;
  }

  // Format the metadata in File for `import Cpp`.
  auto FormatInstRhs(ImportCppDecl /*inst*/) -> void {
    out_ << " ";
    OpenBrace();
    for (ImportCpp import_cpp : sem_ir_->import_cpps().array_ref()) {
      Indent();
      out_ << "import Cpp \""
           << FormatEscaped(
                  sem_ir_->string_literal_values().Get(import_cpp.library_id))
           << "\"\n";
    }
    CloseBrace();
  }

  auto FormatImportRefRhs(ImportIRInstId import_ir_inst_id,
                          EntityNameId entity_name_id,
                          llvm::StringLiteral loaded_label) -> void {
    out_ << " ";
    auto import_ir_inst = sem_ir_->import_ir_insts().Get(import_ir_inst_id);
    FormatArg(import_ir_inst.ir_id);
    out_ << ", ";
    if (entity_name_id.has_value()) {
      // Prefer to show the entity name when possible.
      FormatArg(entity_name_id);
    } else {
      // Show a name based on the location when possible, or the numeric
      // instruction as a last resort.
      const auto& import_ir = sem_ir_->import_irs().Get(import_ir_inst.ir_id);
      auto loc_id = import_ir.sem_ir->insts().GetLocId(import_ir_inst.inst_id);
      if (!loc_id.has_value()) {
        out_ << import_ir_inst.inst_id << " [no loc]";
      } else if (loc_id.is_import_ir_inst_id()) {
        // TODO: Probably don't want to format each indirection, but maybe reuse
        // GetCanonicalImportIRInst?
        out_ << import_ir_inst.inst_id << " [indirect]";
      } else if (loc_id.is_node_id()) {
        // Formats a NodeId from the import.
        const auto& tree = import_ir.sem_ir->parse_tree();
        auto token = tree.node_token(loc_id.node_id());
        out_ << "loc" << tree.tokens().GetLineNumber(token) << "_"
             << tree.tokens().GetColumnNumber(token);
      } else {
        CARBON_FATAL("Unexpected LocId: {0}", loc_id);
      }
    }
    out_ << ", " << loaded_label;
  }

  auto FormatInstRhs(ImportRefLoaded inst) -> void {
    FormatImportRefRhs(inst.import_ir_inst_id, inst.entity_name_id, "loaded");
  }

  auto FormatInstRhs(ImportRefUnloaded inst) -> void {
    FormatImportRefRhs(inst.import_ir_inst_id, inst.entity_name_id, "unloaded");
  }

  auto FormatInstRhs(NameBindingDecl inst) -> void {
    FormatTrailingBlock(inst.pattern_block_id);
  }

  auto FormatInstRhs(SpliceBlock inst) -> void {
    FormatArgs(inst.result_id);
    FormatTrailingBlock(inst.block_id);
  }

  auto FormatInstRhs(WhereExpr inst) -> void {
    FormatArgs(inst.period_self_id);
    FormatTrailingBlock(inst.requirements_id);
  }

  auto FormatInstRhs(StructType inst) -> void {
    out_ << " {";
    llvm::ListSeparator sep;
    for (auto field : sem_ir_->struct_type_fields().Get(inst.fields_id)) {
      out_ << sep << ".";
      FormatName(field.name_id);
      out_ << ": ";
      FormatType(field.type_id);
    }
    out_ << "}";
  }

  auto FormatArgs() -> void {}

  template <typename... Args>
  auto FormatArgs(Args... args) -> void {
    out_ << ' ';
    llvm::ListSeparator sep;
    ((out_ << sep, FormatArg(args)), ...);
  }

  // FormatArg variants handling printing instruction arguments. Several things
  // provide equivalent behavior with `FormatName`, so we provide that as the
  // default.
  template <typename IdT>
  auto FormatArg(IdT id) -> void {
    FormatName(id);
  }

  auto FormatArg(BoolValue v) -> void { out_ << v; }

  auto FormatArg(EntityNameId id) -> void {
    const auto& info = sem_ir_->entity_names().Get(id);
    FormatName(info.name_id);
    if (info.bind_index().has_value()) {
      out_ << ", " << info.bind_index().index;
    }
    if (info.is_template) {
      out_ << ", template";
    }
  }

  auto FormatArg(FacetTypeId id) -> void {
    const auto& info = sem_ir_->facet_types().Get(id);
    // Nothing output to indicate that this is a facet type since this is only
    // used as the argument to a `facet_type` instruction.
    out_ << "<";

    llvm::ListSeparator sep(" & ");
    if (info.impls_constraints.empty()) {
      out_ << "type";
    } else {
      for (auto interface : info.impls_constraints) {
        out_ << sep;
        FormatName(interface.interface_id);
        if (interface.specific_id.has_value()) {
          out_ << ", ";
          FormatName(interface.specific_id);
        }
      }
    }

    if (info.other_requirements || !info.rewrite_constraints.empty()) {
      // TODO: Include specifics.
      out_ << " where ";
      llvm::ListSeparator and_sep(" and ");
      for (auto rewrite : info.rewrite_constraints) {
        out_ << and_sep;
        FormatConstant(rewrite.lhs_const_id);
        out_ << " = ";
        FormatConstant(rewrite.rhs_const_id);
      }
      if (info.other_requirements) {
        out_ << and_sep << "TODO";
      }
    }
    out_ << ">";
  }

  auto FormatArg(IntKind k) -> void { k.Print(out_); }

  auto FormatArg(FloatKind k) -> void { k.Print(out_); }

  auto FormatArg(ImportIRId id) -> void {
    if (id.has_value()) {
      out_ << GetImportIRLabel(id);
    } else {
      out_ << id;
    }
  }

  auto FormatArg(IntId id) -> void {
    // We don't know the signedness to use here. Default to unsigned.
    sem_ir_->ints().Get(id).print(out_, /*isSigned=*/false);
  }

  auto FormatArg(ElementIndex index) -> void { out_ << index; }

  auto FormatArg(CallParamIndex index) -> void { out_ << index; }

  auto FormatArg(NameScopeId id) -> void {
    OpenBrace();
    FormatNameScope(id);
    CloseBrace();
  }

  auto FormatArg(InstBlockId id) -> void {
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

  auto FormatArg(AbsoluteInstBlockId id) -> void {
    FormatArg(static_cast<InstBlockId>(id));
  }

  auto FormatArg(RealId id) -> void {
    // TODO: Format with a `.` when the exponent is near zero.
    const auto& real = sem_ir_->reals().Get(id);
    real.mantissa.print(out_, /*isSigned=*/false);
    out_ << (real.is_decimal ? 'e' : 'p') << real.exponent;
  }

  auto FormatArg(StringLiteralValueId id) -> void {
    out_ << '"'
         << FormatEscaped(sem_ir_->string_literal_values().Get(id),
                          /*use_hex_escapes=*/true)
         << '"';
  }

  auto FormatArg(TypeId id) -> void { FormatType(id); }

  auto FormatArg(TypeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto type_id : sem_ir_->type_blocks().Get(id)) {
      out_ << sep;
      FormatArg(type_id);
    }
    out_ << ')';
  }

  auto FormatReturnSlotArg(InstId dest_id) -> void {
    out_ << " to ";
    FormatArg(dest_id);
  }

  // `FormatName` is used when we need the name from an id. Most id types use
  // equivalent name formatting from InstNamer, although there are a few special
  // formats below.
  template <typename IdT>
  auto FormatName(IdT id) -> void {
    out_ << inst_namer_->GetNameFor(id);
  }

  auto FormatName(NameId id) -> void {
    out_ << sem_ir_->names().GetFormatted(id);
  }

  auto FormatName(InstId id) -> void {
    if (id.has_value()) {
      IncludeChunkInOutput(tentative_inst_chunks_[id.index]);
    }
    out_ << inst_namer_->GetNameFor(scope_, id);
  }

  auto FormatName(AbsoluteInstId id) -> void {
    FormatName(static_cast<InstId>(id));
  }

  auto FormatName(DestInstId id) -> void {
    FormatName(static_cast<InstId>(id));
  }

  auto FormatName(SpecificId id) -> void {
    const auto& specific = sem_ir_->specifics().Get(id);
    FormatName(specific.generic_id);
    FormatArg(specific.args_id);
  }

  auto FormatLabel(InstBlockId id) -> void {
    out_ << inst_namer_->GetLabelFor(scope_, id);
  }

  auto FormatConstant(ConstantId id) -> void {
    if (!id.has_value()) {
      out_ << "<not constant>";
      return;
    }

    // For a symbolic constant in a generic, list the constant value in the
    // generic first, and the canonical constant second.
    if (id.is_symbolic()) {
      const auto& symbolic_constant =
          sem_ir_->constant_values().GetSymbolicConstant(id);
      if (symbolic_constant.generic_id.has_value()) {
        const auto& generic =
            sem_ir_->generics().Get(symbolic_constant.generic_id);
        FormatName(sem_ir_->inst_blocks().Get(generic.GetEvalBlock(
            symbolic_constant.index
                .region()))[symbolic_constant.index.index()]);
        out_ << " (";
        FormatName(sem_ir_->constant_values().GetInstId(id));
        out_ << ")";
        return;
      }
    }

    FormatName(sem_ir_->constant_values().GetInstId(id));
  }

  auto FormatType(TypeId id) -> void {
    if (!id.has_value()) {
      out_ << "invalid";
    } else {
      // Types are formatted in the `constants` scope because they only refer to
      // constants.
      llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::Constants);
      FormatConstant(sem_ir_->types().GetConstantId(id));
    }
  }

  // Returns the label for the indicated IR.
  auto GetImportIRLabel(ImportIRId id) -> std::string {
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

  const File* sem_ir_;
  InstNamer* const inst_namer_;
  Formatter::ShouldFormatEntityFn should_format_entity_;

  // The output stream buffer.
  std::string buffer_;

  // The output stream.
  llvm::raw_string_ostream out_ = llvm::raw_string_ostream(buffer_);

  // Chunks of output text that we have created so far.
  llvm::SmallVector<OutputChunk> output_chunks_;

  // The current scope that we are formatting within. References to names in
  // this scope will not have a `@scope.` prefix added.
  InstNamer::ScopeId scope_ = InstNamer::ScopeId::None;

  // Whether we are formatting in a terminator sequence, that is, a sequence of
  // branches at the end of a block. The entirety of a terminator sequence is
  // formatted on a single line, despite being multiple instructions.
  bool in_terminator_sequence_ = false;

  // The indent depth to use for new instructions.
  int indent_;

  // Whether we are currently formatting immediately after an open brace. If so,
  // a newline will be inserted before the next line indent.
  bool after_open_brace_ = false;

  // The constant value of the current instruction, if it has one that has not
  // yet been printed. The value `NotConstant` is used as a sentinel to indicate
  // there is nothing to print.
  ConstantId pending_constant_value_ = ConstantId::NotConstant;

  // Whether `pending_constant_value_`'s instruction is the same as the
  // instruction currently being printed. If true, only the phase of the
  // constant is printed, and the value is omitted.
  bool pending_constant_value_is_self_ = false;

  // The name of the IR file from which the current entity was imported, if it
  // was imported and no file has been printed yet. This is printed before the
  // first open brace or the semicolon in the entity declaration.
  llvm::StringRef pending_imported_from_;

  // Indexes of chunks of output that should be included when an instruction is
  // referenced, indexed by the instruction's index. This is resized in advance
  // to the correct size.
  llvm::SmallVector<size_t, 0> tentative_inst_chunks_;
};

Formatter::Formatter(const File* sem_ir,
                     ShouldFormatEntityFn should_format_entity)
    : sem_ir_(sem_ir),
      should_format_entity_(should_format_entity),
      inst_namer_(sem_ir) {}

Formatter::~Formatter() = default;

auto Formatter::Print(llvm::raw_ostream& out) -> void {
  FormatterImpl formatter(sem_ir_, &inst_namer_, should_format_entity_,
                          /*indent=*/0);
  formatter.Format();
  formatter.Write(out);
}

}  // namespace Carbon::SemIR

// NOLINTEND(misc-no-recursion)
