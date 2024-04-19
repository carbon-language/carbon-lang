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
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  enum class AddSpace : bool { Before, After };

  explicit Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir,
                     llvm::raw_ostream& out)
      : sem_ir_(sem_ir),
        out_(out),
        inst_namer_(tokenized_buffer, parse_tree, sem_ir) {}

  // Prints the SemIR.
  //
  // Constants are printed first and may be referenced by later sections,
  // including file-scoped instructions. The file scope may contain entity
  // declarations which are defined later, such as classes.
  auto Format() -> void {
    out_ << "--- " << sem_ir_.filename() << "\n\n";

    FormatConstants();

    out_ << inst_namer_.GetScopeName(InstNamer::ScopeId::File) << " ";
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

    // End-of-file newline.
    out_ << "\n";
  }

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

  auto FormatConstants() -> void {
    if (!sem_ir_.constants().size()) {
      return;
    }

    llvm::SaveAndRestore constants_scope(scope_, InstNamer::ScopeId::Constants);
    out_ << inst_namer_.GetScopeName(InstNamer::ScopeId::Constants) << " ";
    OpenBrace();
    FormatCodeBlock(sem_ir_.constants().GetAsVector());
    CloseBrace();
    out_ << "\n\n";
  }

  auto FormatClass(ClassId id) -> void {
    const Class& class_info = sem_ir_.classes().Get(id);

    out_ << "\nclass ";
    FormatClassName(id);

    llvm::SaveAndRestore class_scope(scope_, inst_namer_.GetScopeFor(id));

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
  }

  auto FormatInterface(InterfaceId id) -> void {
    const Interface& interface_info = sem_ir_.interfaces().Get(id);

    out_ << "\ninterface ";
    FormatInterfaceName(id);

    llvm::SaveAndRestore interface_scope(scope_, inst_namer_.GetScopeFor(id));

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
  }

  auto FormatImpl(ImplId id) -> void {
    const Impl& impl_info = sem_ir_.impls().Get(id);

    out_ << "\nimpl ";
    FormatImplName(id);
    out_ << ": ";
    // TODO: Include the deduced parameter list if present.
    FormatType(impl_info.self_id);
    out_ << " as ";
    FormatType(impl_info.constraint_id);

    llvm::SaveAndRestore impl_scope(scope_, inst_namer_.GetScopeFor(id));

    if (impl_info.scope_id.is_valid()) {
      out_ << ' ';
      OpenBrace();
      FormatCodeBlock(impl_info.body_block_id);

      // Print the !members label even if the name scope is empty because we
      // always list the witness in this section.
      IndentLabel();
      out_ << "!members:\n";
      FormatNameScope(impl_info.scope_id);

      Indent();
      out_ << "witness = ";
      FormatArg(impl_info.witness_id);
      out_ << "\n";

      CloseBrace();
      out_ << '\n';
    } else {
      out_ << ";\n";
    }
  }

  auto FormatFunction(FunctionId id) -> void {
    const Function& fn = sem_ir_.functions().Get(id);

    out_ << "\n";

    if (fn.is_extern) {
      out_ << "extern ";
    }

    out_ << "fn ";
    FormatFunctionName(id);

    llvm::SaveAndRestore function_scope(scope_, inst_namer_.GetScopeFor(id));

    if (fn.implicit_param_refs_id != InstBlockId::Empty) {
      out_ << "[";
      FormatParamList(fn.implicit_param_refs_id);
      out_ << "]";
    }

    out_ << "(";
    FormatParamList(fn.param_refs_id);
    out_ << ")";

    if (fn.return_type_id.is_valid()) {
      out_ << " -> ";
      if (!fn.body_block_ids.empty() && fn.has_return_slot()) {
        FormatInstName(fn.return_storage_id);
        out_ << ": ";
      }
      FormatType(fn.return_type_id);
    }

    if (fn.builtin_kind != BuiltinFunctionKind::None) {
      out_ << " = \"";
      out_.write_escaped(fn.builtin_kind.name(),
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
  }

  auto FormatParamList(InstBlockId param_refs_id) -> void {
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
      FormatInstName(param_id);
      out_ << ": ";
      FormatType(sem_ir_.insts().Get(param_id).type_id());
    }
  }

  auto FormatCodeBlock(InstBlockId block_id) -> void {
    if (block_id.is_valid()) {
      FormatCodeBlock(sem_ir_.inst_blocks().Get(block_id));
    }
  }

  auto FormatCodeBlock(llvm::ArrayRef<InstId> block) -> void {
    for (const InstId inst_id : block) {
      FormatInstruction(inst_id);
    }
  }

  auto FormatTrailingBlock(InstBlockId block_id) -> void {
    out_ << ' ';
    OpenBrace();
    FormatCodeBlock(block_id);
    CloseBrace();
  }

  auto FormatNameScope(NameScopeId id, llvm::StringRef label = "") -> void {
    const auto& scope = sem_ir_.name_scopes().Get(id);

    if (scope.names.empty() && scope.extended_scopes.empty() &&
        !scope.has_error) {
      // Name scope is empty.
      return;
    }

    if (!label.empty()) {
      IndentLabel();
      out_ << label;
    }

    // Name scopes aren't kept in any particular order. Sort the entries before
    // we print them for stability and consistency.
    llvm::SmallVector<std::pair<InstId, NameId>> entries;
    for (auto [name_id, inst_id] : scope.names) {
      entries.push_back({inst_id, name_id});
    }
    llvm::sort(entries,
               [](auto a, auto b) { return a.first.index < b.first.index; });

    for (auto [inst_id, name_id] : entries) {
      Indent();
      out_ << ".";
      FormatName(name_id);
      out_ << " = ";
      FormatInstName(inst_id);
      out_ << "\n";
    }

    for (auto extended_scope_id : scope.extended_scopes) {
      // TODO: Print this scope in a better way.
      Indent();
      out_ << "extend " << extended_scope_id << "\n";
    }

    if (scope.has_error) {
      Indent();
      out_ << "has_error\n";
    }
  }

  auto FormatInstruction(InstId inst_id) -> void {
    if (!inst_id.is_valid()) {
      Indent();
      out_ << "invalid\n";
      return;
    }

    FormatInstruction(inst_id, sem_ir_.insts().Get(inst_id));
  }

  auto FormatInstruction(InstId inst_id, Inst inst) -> void {
    CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(InstT)      \
  case CARBON_KIND(InstT typed_inst): {     \
    FormatInstruction(inst_id, typed_inst); \
    break;                                  \
  }
#include "toolchain/sem_ir/inst_kind.def"
    }
  }

  template <typename InstT>
  auto FormatInstruction(InstId inst_id, InstT inst) -> void {
    Indent();
    FormatInstructionLHS(inst_id, inst);
    out_ << InstT::Kind.ir_name();
    pending_constant_value_ = sem_ir_.constant_values().Get(inst_id);
    pending_constant_value_is_self_ =
        pending_constant_value_.inst_id() == inst_id;
    FormatInstructionRHS(inst);
    FormatPendingConstantValue(AddSpace::Before);
    out_ << "\n";
  }

  // Don't print a constant for ImportRefUnloaded.
  auto FormatInstruction(InstId inst_id, ImportRefUnloaded inst) -> void {
    Indent();
    FormatInstructionLHS(inst_id, inst);
    out_ << ImportRefUnloaded::Kind.ir_name();
    FormatInstructionRHS(inst);
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
        FormatInstName(pending_constant_value_.inst_id());
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

  auto FormatInstructionLHS(InstId inst_id, Inst inst) -> void {
    switch (inst.kind().value_kind()) {
      case InstValueKind::Typed:
        FormatInstName(inst_id);
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

  // Print ImportRefUnloaded with type-like semantics even though it lacks a
  // type_id.
  auto FormatInstructionLHS(InstId inst_id, ImportRefUnloaded /*inst*/)
      -> void {
    FormatInstName(inst_id);
    out_ << " = ";
  }

  template <typename InstT>
  auto FormatInstructionRHS(InstT inst) -> void {
    // By default, an instruction has a comma-separated argument list.
    using Info = Internal::InstLikeTypeInfo<InstT>;
    if constexpr (Info::NumArgs == 2) {
      FormatArgs(Info::template Get<0>(inst), Info::template Get<1>(inst));
    } else if constexpr (Info::NumArgs == 1) {
      FormatArgs(Info::template Get<0>(inst));
    } else {
      FormatArgs();
    }
  }

  auto FormatInstructionRHS(BindSymbolicName inst) -> void {
    // A BindSymbolicName with no value is a purely symbolic binding, such as
    // the `Self` in an interface. Don't print out `invalid` for the value.
    if (inst.value_id.is_valid()) {
      FormatArgs(inst.bind_name_id, inst.value_id);
    } else {
      FormatArgs(inst.bind_name_id);
    }
  }

  auto FormatInstructionRHS(BlockArg inst) -> void {
    out_ << " ";
    FormatLabel(inst.block_id);
  }

  auto FormatInstructionRHS(Namespace inst) -> void {
    if (inst.import_id.is_valid()) {
      FormatArgs(inst.import_id, inst.name_scope_id);
    } else {
      FormatArgs(inst.name_scope_id);
    }
  }

  auto FormatInstruction(InstId /*inst_id*/, BranchIf inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << "if ";
    FormatInstName(inst.cond_id);
    out_ << " " << Branch::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << " else ";
    in_terminator_sequence_ = true;
  }

  auto FormatInstruction(InstId /*inst_id*/, BranchWithArg inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << BranchWithArg::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << "(";
    FormatInstName(inst.arg_id);
    out_ << ")\n";
    in_terminator_sequence_ = false;
  }

  auto FormatInstruction(InstId /*inst_id*/, Branch inst) -> void {
    if (!in_terminator_sequence_) {
      Indent();
    }
    out_ << Branch::Kind.ir_name() << " ";
    FormatLabel(inst.target_id);
    out_ << "\n";
    in_terminator_sequence_ = false;
  }

  auto FormatInstructionRHS(Call inst) -> void {
    out_ << " ";
    FormatArg(inst.callee_id);

    if (!inst.args_id.is_valid()) {
      out_ << "(<invalid>)";
      return;
    }

    llvm::ArrayRef<InstId> args = sem_ir_.inst_blocks().Get(inst.args_id);

    bool has_return_slot = GetInitRepr(sem_ir_, inst.type_id).has_return_slot();
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

  auto FormatInstructionRHS(ArrayInit inst) -> void {
    FormatArgs(inst.inits_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstructionRHS(InitializeFrom inst) -> void {
    FormatArgs(inst.src_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstructionRHS(StructInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlot(init.dest_id);
  }

  auto FormatInstructionRHS(TupleInit init) -> void {
    FormatArgs(init.elements_id);
    FormatReturnSlot(init.dest_id);
  }

  auto FormatInstructionRHS(FunctionDecl inst) -> void {
    FormatArgs(inst.function_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(ClassDecl inst) -> void {
    FormatArgs(inst.class_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(ImplDecl inst) -> void {
    FormatArgs(inst.impl_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(InterfaceDecl inst) -> void {
    FormatArgs(inst.interface_id);
    FormatTrailingBlock(inst.decl_block_id);
  }

  auto FormatInstructionRHS(IntLiteral inst) -> void {
    out_ << " ";
    sem_ir_.ints()
        .Get(inst.int_id)
        .print(out_, sem_ir_.types().IsSignedInt(inst.type_id));
  }

  auto FormatInstructionRHS(ImportRefUnloaded inst) -> void {
    FormatArgs(inst.import_ir_inst_id);
    out_ << ", unloaded";
  }

  auto FormatInstructionRHS(ImportRefLoaded inst) -> void {
    FormatArgs(inst.import_ir_inst_id);
    out_ << ", loaded";
  }

  auto FormatInstructionRHS(ImportRefUsed inst) -> void {
    FormatArgs(inst.import_ir_inst_id, inst.used_id);
  }

  auto FormatInstructionRHS(SpliceBlock inst) -> void {
    FormatArgs(inst.result_id);
    FormatTrailingBlock(inst.block_id);
  }

  // StructTypeFields are formatted as part of their StructType.
  auto FormatInstruction(InstId /*inst_id*/, StructTypeField /*inst*/) -> void {
  }

  auto FormatInstructionRHS(StructType inst) -> void {
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

  auto FormatArg(BoolValue v) -> void { out_ << v; }

  auto FormatArg(BuiltinKind kind) -> void { out_ << kind.label(); }

  auto FormatArg(BindNameId id) -> void {
    FormatName(sem_ir_.bind_names().Get(id).name_id);
  }

  auto FormatArg(FunctionId id) -> void { FormatFunctionName(id); }

  auto FormatArg(ClassId id) -> void { FormatClassName(id); }

  auto FormatArg(InterfaceId id) -> void { FormatInterfaceName(id); }

  auto FormatArg(IntKind k) -> void { k.Print(out_); }

  auto FormatArg(ImplId id) -> void { FormatImplName(id); }

  auto FormatArg(ImportIRId id) -> void { out_ << id; }

  auto FormatArg(ImportIRInstId id) -> void {
    // Don't format the inst_id because it refers to a different IR.
    // TODO: Consider a better way to format the InstID from other IRs.
    auto import_ir_inst = sem_ir_.import_ir_insts().Get(id);
    out_ << import_ir_inst.ir_id << ", " << import_ir_inst.inst_id;
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

  auto FormatArg(NameScopeId id) -> void {
    OpenBrace();
    FormatNameScope(id);
    CloseBrace();
  }

  auto FormatArg(InstId id) -> void { FormatInstName(id); }

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

  auto FormatArg(NameId id) -> void { FormatName(id); }

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

  auto FormatName(NameId id) -> void {
    out_ << sem_ir_.names().GetFormatted(id);
  }

  auto FormatInstName(InstId id) -> void {
    out_ << inst_namer_.GetNameFor(scope_, id);
  }

  auto FormatLabel(InstBlockId id) -> void {
    out_ << inst_namer_.GetLabelFor(scope_, id);
  }

  auto FormatFunctionName(FunctionId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatClassName(ClassId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatInterfaceName(InterfaceId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatImplName(ImplId id) -> void { out_ << inst_namer_.GetNameFor(id); }

  auto FormatType(TypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << sem_ir_.StringifyType(id);
    }
  }

 private:
  const File& sem_ir_;
  llvm::raw_ostream& out_;
  InstNamer inst_namer_;

  // The current scope that we are formatting within. References to names in
  // this scope will not have a `@scope.` prefix added.
  InstNamer::ScopeId scope_ = InstNamer::ScopeId::None;

  // Whether we are formatting in a terminator sequence, that is, a sequence of
  // branches at the end of a block. The entirety of a terminator sequence is
  // formatted on a single line, despite being multiple instructions.
  bool in_terminator_sequence_ = false;

  // The indent depth to use for new instructions.
  int indent_ = 0;

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

auto FormatFile(const Lex::TokenizedBuffer& tokenized_buffer,
                const Parse::Tree& parse_tree, const File& sem_ir,
                llvm::raw_ostream& out) -> void {
  Formatter(tokenized_buffer, parse_tree, sem_ir, out).Format();
}

}  // namespace Carbon::SemIR
