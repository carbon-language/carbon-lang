// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter.h"

#include "common/ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_namer.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class FormatterImpl {
 public:
  explicit FormatterImpl(const File& sem_ir, InstNamer* inst_namer,
                         llvm::raw_ostream& out, int indent)
      : sem_ir_(sem_ir), inst_namer_(inst_namer), out_(out), indent_(indent) {}

  // Prints the SemIR.
  //
  // Constants are printed first and may be referenced by later sections,
  // including file-scoped instructions. The file scope may contain entity
  // declarations which are defined later, such as classes.
  auto Format() -> void {
    out_ << "--- " << sem_ir_.filename() << "\n\n";

    FormatScope(InstNamer::ScopeId::Constants, sem_ir_.constants().array_ref());
    FormatScope(InstNamer::ScopeId::ImportRefs,
                sem_ir_.inst_blocks().Get(InstBlockId::ImportRefs));

    out_ << inst_namer_->GetScopeName(InstNamer::ScopeId::File) << " ";
    OpenBrace();

    // TODO: Handle the case where there are multiple top-level instruction
    // blocks. For example, there may be branching in the initializer of a
    // global or a type expression.
    if (auto block_id = sem_ir_.top_inst_block_id(); block_id.is_valid()) {
      llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::File);
      FormatCodeBlock(block_id);
    }

    CloseBrace();
    out_ << '\n';

    for (int i : llvm::seq(sem_ir_.interfaces().size())) {
      FormatInterface(InterfaceId(i));
    }

    for (int i : llvm::seq(sem_ir_.impls().size())) {
      FormatImpl(ImplId(i));
    }

    for (int i : llvm::seq(sem_ir_.classes().size())) {
      FormatClass(ClassId(i));
    }

    for (int i : llvm::seq(sem_ir_.functions().size())) {
      FormatFunction(FunctionId(i));
    }

    for (int i : llvm::seq(sem_ir_.specifics().size())) {
      FormatSpecific(SpecificId(i));
    }

    // End-of-file newline.
    out_ << "\n";
  }

  // Prints a code block.
  auto FormatPartialTrailingCodeBlock(llvm::ArrayRef<SemIR::InstId> block)
      -> void {
    out_ << ' ';
    OpenBrace();
    constexpr int NumPrintedOnSkip = 9;
    // Avoid only skipping one item.
    if (block.size() > NumPrintedOnSkip + 1) {
      Indent();
      out_ << "... skipping " << (block.size() - NumPrintedOnSkip)
           << " insts ...\n";
      block = block.take_back(NumPrintedOnSkip);
    }
    FormatCodeBlock(block);
    CloseBrace();
  }

  // Prints a single instruction.
  auto FormatInst(InstId inst_id) -> void {
    if (!inst_id.is_valid()) {
      Indent();
      out_ << "invalid\n";
      return;
    }

    FormatInst(inst_id, sem_ir_.insts().Get(inst_id));
  }

 private:
  enum class AddSpace : bool { Before, After };

  // Begins a braced block. Writes an open brace, and prepares to insert a
  // newline after it if the braced block is non-empty.
  auto OpenBrace() -> void {
    // Put the constant value of an instruction before any braced block, rather
    // than at the end.
    FormatPendingConstantValue(AddSpace::After);

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

  // Formats a top-level scope, particularly Constants and ImportRefs.
  auto FormatScope(InstNamer::ScopeId scope_id, llvm::ArrayRef<InstId> block)
      -> void {
    if (block.empty()) {
      return;
    }

    llvm::SaveAndRestore scope(scope_, scope_id);
    out_ << inst_namer_->GetScopeName(scope_id) << " ";
    OpenBrace();
    FormatCodeBlock(block);
    CloseBrace();
    out_ << "\n\n";
  }

  // Formats a full class.
  auto FormatClass(ClassId id) -> void {
    const Class& class_info = sem_ir_.classes().Get(id);
    FormatEntityStart("class", class_info.generic_id, id);

    llvm::SaveAndRestore class_scope(scope_, inst_namer_->GetScopeFor(id));

    if (class_info.scope_id.is_valid()) {
      out_ << ' ';
      OpenBrace();
      FormatCodeBlock(class_info.body_block_id);
      FormatNameScope(class_info.scope_id, "!members:\n");
      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }

    FormatEntityEnd(class_info.generic_id);
  }

  // Formats a full interface.
  auto FormatInterface(InterfaceId id) -> void {
    const Interface& interface_info = sem_ir_.interfaces().Get(id);
    FormatEntityStart("interface", interface_info.generic_id, id);

    llvm::SaveAndRestore interface_scope(scope_, inst_namer_->GetScopeFor(id));

    if (interface_info.scope_id.is_valid()) {
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
      out_ << '\n';
    } else {
      out_ << ";\n";
    }

    FormatEntityEnd(interface_info.generic_id);
  }

  // Formats a full impl.
  auto FormatImpl(ImplId id) -> void {
    const Impl& impl_info = sem_ir_.impls().Get(id);
    FormatEntityStart("impl", impl_info.generic_id, id);

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
      if (impl_info.scope_id.is_valid()) {
        FormatNameScope(impl_info.scope_id);
      }

      Indent();
      out_ << "witness = ";
      FormatArg(impl_info.witness_id);
      out_ << "\n";

      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }

    FormatEntityEnd(impl_info.generic_id);
  }

  // Formats a full function.
  auto FormatFunction(FunctionId id) -> void {
    const Function& fn = sem_ir_.functions().Get(id);
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
    FormatEntityStart(function_start, fn.generic_id, id);

    llvm::SaveAndRestore function_scope(scope_, inst_namer_->GetScopeFor(id));

    FormatParamList(fn.implicit_param_refs_id, /*is_implicit=*/true);
    FormatParamList(fn.param_refs_id, /*is_implicit=*/false);

    if (fn.return_storage_id.is_valid()) {
      out_ << " -> ";
      auto return_info = ReturnTypeInfo::ForFunction(sem_ir_, fn);
      if (!fn.body_block_ids.empty() && return_info.is_valid() &&
          return_info.has_return_slot()) {
        FormatName(fn.return_storage_id);
        out_ << ": ";
      }
      FormatType(sem_ir_.insts().Get(fn.return_storage_id).type_id());
    }

    if (fn.builtin_function_kind != BuiltinFunctionKind::None) {
      out_ << " = \"";
      out_.write_escaped(fn.builtin_function_kind.name(),
                         /*UseHexEscapes=*/true);
      out_ << "\"";
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
      out_ << '\n';
    } else {
      out_ << ";\n";
    }

    FormatEntityEnd(fn.generic_id);
  }

  // Helper for FormatSpecific to print regions.
  auto FormatSpecificRegion(const Generic& generic, const Specific& specific,
                            GenericInstIndex::Region region,
                            llvm::StringRef region_name) -> void {
    if (!specific.GetValueBlock(region).is_valid()) {
      return;
    }

    if (!region_name.empty()) {
      IndentLabel();
      out_ << "!" << region_name << ":\n";
    }
    for (auto [generic_inst_id, specific_inst_id] : llvm::zip_longest(
             sem_ir_.inst_blocks().GetOrEmpty(generic.GetEvalBlock(region)),
             sem_ir_.inst_blocks().GetOrEmpty(
                 specific.GetValueBlock(region)))) {
      if (generic_inst_id && specific_inst_id &&
          sem_ir_.insts().Is<StructTypeField>(*generic_inst_id) &&
          sem_ir_.insts().Is<StructTypeField>(*specific_inst_id)) {
        // Skip printing struct type fields to match the way we print the
        // generic.
        continue;
      }

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
    const auto& specific = sem_ir_.specifics().Get(id);

    out_ << "\n";

    out_ << "specific ";
    FormatName(id);

    // TODO: Remove once we stop forming generic specifics with no generic
    // during import.
    if (!specific.generic_id.is_valid()) {
      out_ << ";\n";
      return;
    }
    out_ << " ";

    const auto& generic = sem_ir_.generics().Get(specific.generic_id);
    llvm::SaveAndRestore generic_scope(
        scope_, inst_namer_->GetScopeFor(specific.generic_id));

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
    const auto& generic = sem_ir_.generics().Get(generic_id);
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
    if (generic.definition_block_id.is_valid()) {
      IndentLabel();
      out_ << "!definition:\n";
      FormatCodeBlock(generic.definition_block_id);
    }
  }

  // Provides common formatting for entities, paired with FormatEntityEnd.
  template <typename IdT>
  auto FormatEntityStart(llvm::StringRef entity_kind, GenericId generic_id,
                         IdT entity_id) -> void {
    if (generic_id.is_valid()) {
      FormatGenericStart(entity_kind, generic_id);
    }

    out_ << "\n";
    Indent();
    out_ << entity_kind;

    // If there's a generic, it will have attached the name. Otherwise, add the
    // name here.
    if (!generic_id.is_valid()) {
      out_ << " ";
      FormatName(entity_id);
    }
  }

  // Provides common formatting for entities, paired with FormatEntityStart.
  auto FormatEntityEnd(GenericId generic_id) -> void {
    if (generic_id.is_valid()) {
      CloseBrace();
      out_ << '\n';
    }
  }

  // Formats parameters, eliding them completely if they're empty. Wraps in
  // parentheses or square brackets based on whether these are implicit
  // parameters.
  auto FormatParamList(InstBlockId param_refs_id, bool is_implicit) -> void {
    if (!param_refs_id.is_valid()) {
      return;
    }

    out_ << (is_implicit ? "[" : "(");

    llvm::ListSeparator sep;
    for (InstId param_id : sem_ir_.inst_blocks().Get(param_refs_id)) {
      out_ << sep;
      if (!param_id.is_valid()) {
        out_ << "invalid";
        continue;
      }
      if (auto addr = sem_ir_.insts().TryGetAs<SemIR::AddrPattern>(param_id)) {
        out_ << "addr ";
        param_id = addr->inner_id;
      }
      FormatName(param_id);
      out_ << ": ";
      FormatType(sem_ir_.insts().Get(param_id).type_id());
    }

    out_ << (is_implicit ? "]" : ")");
  }

  // Prints instructions for a code block.
  auto FormatCodeBlock(InstBlockId block_id) -> void {
    if (block_id.is_valid()) {
      FormatCodeBlock(sem_ir_.inst_blocks().Get(block_id));
    }
  }

  // Prints instructions for a code block.
  auto FormatCodeBlock(llvm::ArrayRef<InstId> block) -> void {
    for (const InstId inst_id : block) {
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
    const auto& scope = sem_ir_.name_scopes().Get(id);

    if (scope.names.empty() && scope.extended_scopes.empty() &&
        scope.import_ir_scopes.empty() && !scope.has_error) {
      // Name scope is empty.
      return;
    }

    if (!label.empty()) {
      IndentLabel();
      out_ << label;
    }

    for (auto [name_id, inst_id, access_kind] : scope.names) {
      Indent();
      out_ << ".";
      FormatName(name_id);
      switch (access_kind) {
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
      FormatName(inst_id);
      out_ << "\n";
    }

    for (auto extended_scope_id : scope.extended_scopes) {
      // TODO: Print this scope in a better way.
      Indent();
      out_ << "extend " << extended_scope_id << "\n";
    }

    for (auto [import_ir_id, unused] : scope.import_ir_scopes) {
      Indent();
      out_ << "import ";
      FormatArg(import_ir_id);
      out_ << "\n";
    }

    if (scope.has_error) {
      Indent();
      out_ << "has_error\n";
    }
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
    FormatInstLHS(inst_id, inst);
    out_ << InstT::Kind.ir_name();
    pending_constant_value_ = sem_ir_.constant_values().Get(inst_id);
    pending_constant_value_is_self_ =
        sem_ir_.constant_values().GetInstIdIfValid(pending_constant_value_) ==
        inst_id;
    FormatInstRHS(inst);
    FormatPendingConstantValue(AddSpace::Before);
    out_ << "\n";
  }

  // Don't print a constant for ImportRefUnloaded.
  auto FormatInst(InstId inst_id, ImportRefUnloaded inst) -> void {
    Indent();
    FormatInstLHS(inst_id, inst);
    out_ << ImportRefUnloaded::Kind.ir_name();
    FormatInstRHS(inst);
    out_ << "\n";
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
    if (pending_constant_value_.is_valid()) {
      out_ << (pending_constant_value_.is_symbolic() ? "symbolic" : "template");
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

  auto FormatInstLHS(InstId inst_id, Inst inst) -> void {
    switch (inst.kind().value_kind()) {
      case InstValueKind::Typed:
        FormatName(inst_id);
        out_ << ": ";
        switch (GetExprCategory(sem_ir_, inst_id)) {
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

  // Format ImportDecl with its name.
  auto FormatInstLHS(InstId inst_id, ImportDecl /*inst*/) -> void {
    FormatName(inst_id);
    out_ << " = ";
  }

  // Print ImportRefUnloaded with type-like semantics even though it lacks a
  // type_id.
  auto FormatInstLHS(InstId inst_id, ImportRefUnloaded /*inst*/) -> void {
    FormatName(inst_id);
    out_ << " = ";
  }

  template <typename InstT>
  auto FormatInstRHS(InstT inst) -> void {
    // By default, an instruction has a comma-separated argument list.
    using Info = Internal::InstLikeTypeInfo<InstT>;
    if constexpr (Info::NumArgs == 2) {
      // Several instructions have a second operand that's a specific ID. We
      // don't include it in the argument list if there is no corresponding
      // specific, that is, when we're not in a generic context.
      if constexpr (std::is_same_v<typename Info::template ArgType<1>,
                                   SemIR::SpecificId>) {
        if (!Info::template Get<1>(inst).is_valid()) {
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

  auto FormatInstRHS(BindSymbolicName inst) -> void {
    // A BindSymbolicName with no value is a purely symbolic binding, such as
    // the `Self` in an interface. Don't print out `invalid` for the value.
    if (inst.value_id.is_valid()) {
      FormatArgs(inst.entity_name_id, inst.value_id);
    } else {
      FormatArgs(inst.entity_name_id);
    }
  }

  auto FormatInstRHS(BlockArg inst) -> void {
    out_ << " ";
    FormatLabel(inst.block_id);
  }

  auto FormatInstRHS(Namespace inst) -> void {
    if (inst.import_id.is_valid()) {
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

  auto FormatInstRHS(Call inst) -> void {
    out_ << " ";
    FormatArg(inst.callee_id);

    if (!inst.args_id.is_valid()) {
      out_ << "(<invalid>)";
      return;
    }

    llvm::ArrayRef<InstId> args = sem_ir_.inst_blocks().Get(inst.args_id);

    auto return_info = ReturnTypeInfo::ForType(sem_ir_, inst.type_id);
    bool has_return_slot = return_info.has_return_slot();
    InstId return_slot_id = InstId::Invalid;
    if (has_return_slot) {
      return_slot_id = args.back();
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
      FormatReturnSlot(return_slot_id);
    }
  }

  auto FormatInstRHS(ArrayInit inst) -> void {
    FormatArgs(inst.inits_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstRHS(InitializeFrom inst) -> void {
    FormatArgs(inst.src_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstRHS(ReturnExpr ret) -> void {
    FormatArgs(ret.expr_id);
    if (ret.dest_id.is_valid()) {
      FormatReturnSlot(ret.dest_id);
    }
  }

  auto FormatInstRHS(StructInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlot(init.dest_id);
  }

  auto FormatInstRHS(TupleInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlot(init.dest_id);
  }

  auto FormatInstRHS(FunctionDecl inst) -> void {
    FormatArgs(inst.function_id);
    llvm::SaveAndRestore class_scope(
        scope_, inst_namer_->GetScopeFor(inst.function_id));
    FormatTrailingBlock(
        sem_ir_.functions().Get(inst.function_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRHS(ClassDecl inst) -> void {
    FormatArgs(inst.class_id);
    llvm::SaveAndRestore class_scope(scope_,
                                     inst_namer_->GetScopeFor(inst.class_id));
    FormatTrailingBlock(sem_ir_.classes().Get(inst.class_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRHS(ImplDecl inst) -> void {
    FormatArgs(inst.impl_id);
    llvm::SaveAndRestore class_scope(scope_,
                                     inst_namer_->GetScopeFor(inst.impl_id));
    FormatTrailingBlock(sem_ir_.impls().Get(inst.impl_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRHS(InterfaceDecl inst) -> void {
    FormatArgs(inst.interface_id);
    llvm::SaveAndRestore class_scope(
        scope_, inst_namer_->GetScopeFor(inst.interface_id));
    FormatTrailingBlock(
        sem_ir_.interfaces().Get(inst.interface_id).pattern_block_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstRHS(IntLiteral inst) -> void {
    out_ << " ";
    sem_ir_.ints()
        .Get(inst.int_id)
        .print(out_, sem_ir_.types().IsSignedInt(inst.type_id));
  }

  auto FormatInstRHS(FloatLiteral inst) -> void {
    llvm::SmallVector<char, 16> buffer;
    sem_ir_.floats().Get(inst.float_id).toString(buffer);
    out_ << " " << buffer;
  }

  auto FormatInstRHS(ImportRefUnloaded inst) -> void {
    FormatArgs(inst.import_ir_inst_id);
    out_ << ", unloaded";
  }

  auto FormatInstRHS(ImportRefLoaded inst) -> void {
    FormatArgs(inst.import_ir_inst_id);
    out_ << ", loaded";
  }

  auto FormatInstRHS(SpliceBlock inst) -> void {
    FormatArgs(inst.result_id);
    FormatTrailingBlock(inst.block_id);
  }

  auto FormatInstRHS(WhereExpr inst) -> void {
    FormatArgs(inst.period_self_id);
    FormatTrailingBlock(inst.requirements_id);
  }

  // StructTypeFields are formatted as part of their StructType.
  auto FormatInst(InstId /*inst_id*/, StructTypeField /*inst*/) -> void {}

  auto FormatInstRHS(StructType inst) -> void {
    out_ << " {";
    llvm::ListSeparator sep;
    for (auto field_id : sem_ir_.inst_blocks().Get(inst.fields_id)) {
      out_ << sep << ".";
      auto field = sem_ir_.insts().GetAs<StructTypeField>(field_id);
      FormatName(field.name_id);
      out_ << ": ";
      FormatType(field.field_type_id);
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

  auto FormatArg(BuiltinInstKind kind) -> void { out_ << kind.label(); }

  auto FormatArg(EntityNameId id) -> void {
    const auto& info = sem_ir_.entity_names().Get(id);
    FormatName(info.name_id);
    if (info.bind_index.is_valid()) {
      out_ << ", " << info.bind_index.index;
    }
  }

  auto FormatArg(IntKind k) -> void { k.Print(out_); }

  auto FormatArg(FloatKind k) -> void { k.Print(out_); }

  auto FormatArg(ImportIRId id) -> void {
    if (!id.is_valid()) {
      out_ << id;
      return;
    }
    const auto& import_ir = *sem_ir_.import_irs().Get(id).sem_ir;
    if (import_ir.package_id().is_valid()) {
      out_ << import_ir.identifiers().Get(import_ir.package_id());
    } else {
      out_ << "Main";
    }
    out_ << "//";
    CARBON_CHECK(import_ir.library_id().is_valid());
    if (import_ir.library_id() == LibraryNameId::Default) {
      out_ << "default";
    } else {
      out_ << import_ir.string_literal_values().Get(
          import_ir.library_id().AsStringLiteralValueId());
    }
  }

  auto FormatArg(ImportIRInstId id) -> void {
    // Don't format the inst_id because it refers to a different IR.
    // TODO: Consider a better way to format the InstID from other IRs.
    auto import_ir_inst = sem_ir_.import_ir_insts().Get(id);
    FormatArg(import_ir_inst.ir_id);
    out_ << ", " << import_ir_inst.inst_id;
  }

  auto FormatArg(IntId id) -> void {
    // We don't know the signedness to use here. Default to unsigned.
    sem_ir_.ints().Get(id).print(out_, /*isSigned=*/false);
  }

  auto FormatArg(LocId id) -> void {
    if (id.is_import_ir_inst_id()) {
      out_ << "{";
      FormatArg(id.import_ir_inst_id());
      out_ << "}";
    } else {
      // TODO: For a NodeId, this prints the index of the node. Do we want it to
      // print a line number or something in order to make it less dependent on
      // parse?
      out_ << id;
    }
  }

  auto FormatArg(ElementIndex index) -> void { out_ << index; }

  auto FormatArg(RuntimeParamIndex index) -> void { out_ << index; }

  auto FormatArg(NameScopeId id) -> void {
    OpenBrace();
    FormatNameScope(id);
    CloseBrace();
  }

  auto FormatArg(InstBlockId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
      return;
    }

    out_ << '(';
    llvm::ListSeparator sep;
    for (auto inst_id : sem_ir_.inst_blocks().Get(id)) {
      out_ << sep;
      FormatArg(inst_id);
    }
    out_ << ')';
  }

  auto FormatArg(RealId id) -> void {
    // TODO: Format with a `.` when the exponent is near zero.
    const auto& real = sem_ir_.reals().Get(id);
    real.mantissa.print(out_, /*isSigned=*/false);
    out_ << (real.is_decimal ? 'e' : 'p') << real.exponent;
  }

  auto FormatArg(StringLiteralValueId id) -> void {
    out_ << '"';
    out_.write_escaped(sem_ir_.string_literal_values().Get(id),
                       /*UseHexEscapes=*/true);
    out_ << '"';
  }

  auto FormatArg(TypeId id) -> void { FormatType(id); }

  auto FormatArg(TypeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto type_id : sem_ir_.type_blocks().Get(id)) {
      out_ << sep;
      FormatArg(type_id);
    }
    out_ << ')';
  }

  auto FormatReturnSlot(InstId dest_id) -> void {
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
    out_ << sem_ir_.names().GetFormatted(id);
  }

  auto FormatName(InstId id) -> void {
    out_ << inst_namer_->GetNameFor(scope_, id);
  }

  auto FormatName(AbsoluteInstId id) -> void {
    FormatName(static_cast<InstId>(id));
  }

  auto FormatName(SpecificId id) -> void {
    const auto& specific = sem_ir_.specifics().Get(id);
    FormatName(specific.generic_id);
    FormatArg(specific.args_id);
  }

  auto FormatLabel(InstBlockId id) -> void {
    out_ << inst_namer_->GetLabelFor(scope_, id);
  }

  auto FormatConstant(ConstantId id) -> void {
    if (!id.is_valid()) {
      out_ << "<not constant>";
      return;
    }

    // For a symbolic constant in a generic, list the constant value in the
    // generic first, and the canonical constant second.
    if (id.is_symbolic()) {
      const auto& symbolic_constant =
          sem_ir_.constant_values().GetSymbolicConstant(id);
      if (symbolic_constant.generic_id.is_valid()) {
        const auto& generic =
            sem_ir_.generics().Get(symbolic_constant.generic_id);
        FormatName(sem_ir_.inst_blocks().Get(generic.GetEvalBlock(
            symbolic_constant.index
                .region()))[symbolic_constant.index.index()]);
        out_ << " (";
        FormatName(sem_ir_.constant_values().GetInstId(id));
        out_ << ")";
        return;
      }
    }

    FormatName(sem_ir_.constant_values().GetInstId(id));
  }

  auto FormatType(TypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      // Types are formatted in the `constants` scope because they only refer to
      // constants.
      llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeId::Constants);
      FormatConstant(sem_ir_.types().GetConstantId(id));
    }
  }

  const File& sem_ir_;
  InstNamer* const inst_namer_;

  // The output stream. Set while formatting instructions.
  llvm::raw_ostream& out_;

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
};

Formatter::Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir)
    : sem_ir_(sem_ir), inst_namer_(tokenized_buffer, parse_tree, sem_ir) {}

Formatter::~Formatter() = default;

auto Formatter::Print(llvm::raw_ostream& out) -> void {
  FormatterImpl formatter(sem_ir_, &inst_namer_, out, /*indent=*/0);
  formatter.Format();
}

auto Formatter::PrintPartialTrailingCodeBlock(
    llvm::ArrayRef<SemIR::InstId> block, int indent, llvm::raw_ostream& out)
    -> void {
  FormatterImpl formatter(sem_ir_, &inst_namer_, out, indent);
  formatter.FormatPartialTrailingCodeBlock(block);
}

auto Formatter::PrintInst(SemIR::InstId inst_id, int indent,
                          llvm::raw_ostream& out) -> void {
  FormatterImpl formatter(sem_ir_, &inst_namer_, out, indent);
  formatter.FormatInst(inst_id);
}

}  // namespace Carbon::SemIR
