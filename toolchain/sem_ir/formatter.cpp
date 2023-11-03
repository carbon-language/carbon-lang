// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/base/value_store.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"

namespace Carbon::SemIR {

namespace {
// Assigns names to instructions, blocks, and scopes in the Semantics IR.
//
// TODOs / future work ideas:
// - Add a documentation file for the textual format and link to the
//   naming section here.
// - Consider representing literals as just `literal` in the IR and using the
//   type to distinguish.
class InstNamer {
 public:
  // int32_t matches the input value size.
  // NOLINTNEXTLINE(performance-enum-size)
  enum class ScopeIndex : int32_t {
    None = -1,
    File = 0,
  };
  static_assert(sizeof(ScopeIndex) == sizeof(FunctionId));

  InstNamer(const Lex::TokenizedBuffer& tokenized_buffer,
            const Parse::Tree& parse_tree, const File& sem_ir)
      : tokenized_buffer_(tokenized_buffer),
        parse_tree_(parse_tree),
        sem_ir_(sem_ir) {
    insts.resize(sem_ir.insts().size());
    labels.resize(sem_ir.inst_blocks().size());
    scopes.resize(1 + sem_ir.functions().size() + sem_ir.classes().size());

    // Build the file scope.
    GetScopeInfo(ScopeIndex::File).name = globals.AddNameUnchecked("file");
    CollectNamesInBlock(ScopeIndex::File, sem_ir.top_inst_block_id());

    // Build each function scope.
    for (auto [i, fn] : llvm::enumerate(sem_ir.functions().array_ref())) {
      auto fn_id = FunctionId(i);
      auto fn_scope = GetScopeFor(fn_id);
      // TODO: Provide a location for the function for use as a
      // disambiguator.
      auto fn_loc = Parse::Node::Invalid;
      GetScopeInfo(fn_scope).name = globals.AllocateName(
          *this, fn_loc,
          fn.name_id.is_valid() ? sem_ir.identifiers().Get(fn.name_id).str()
                                : "");
      CollectNamesInBlock(fn_scope, fn.implicit_param_refs_id);
      CollectNamesInBlock(fn_scope, fn.param_refs_id);
      if (fn.return_slot_id.is_valid()) {
        insts[fn.return_slot_id.index] = {
            fn_scope,
            GetScopeInfo(fn_scope).insts.AllocateName(
                *this, sem_ir.insts().Get(fn.return_slot_id).parse_node(),
                "return")};
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
    }

    // Build each class scope.
    for (auto [i, class_info] : llvm::enumerate(sem_ir.classes().array_ref())) {
      auto class_id = ClassId(i);
      auto class_scope = GetScopeFor(class_id);
      // TODO: Provide a location for the class for use as a
      // disambiguator.
      auto class_loc = Parse::Node::Invalid;
      GetScopeInfo(class_scope).name = globals.AllocateName(
          *this, class_loc,
          class_info.name_id.is_valid()
              ? sem_ir.identifiers().Get(class_info.name_id).str()
              : "");
      AddBlockLabel(class_scope, class_info.body_block_id, "class", class_loc);
      CollectNamesInBlock(class_scope, class_info.body_block_id);
    }
  }

  // Returns the scope index corresponding to a function.
  auto GetScopeFor(FunctionId fn_id) -> ScopeIndex {
    return static_cast<ScopeIndex>(1 + fn_id.index);
  }

  // Returns the scope index corresponding to a class.
  auto GetScopeFor(ClassId class_id) -> ScopeIndex {
    return static_cast<ScopeIndex>(1 + sem_ir_.functions().size() +
                                   class_id.index);
  }

  // Returns the IR name to use for a function.
  auto GetNameFor(FunctionId fn_id) -> llvm::StringRef {
    if (!fn_id.is_valid()) {
      return "invalid";
    }
    return GetScopeInfo(GetScopeFor(fn_id)).name.str();
  }

  // Returns the IR name to use for a class.
  auto GetNameFor(ClassId class_id) -> llvm::StringRef {
    if (!class_id.is_valid()) {
      return "invalid";
    }
    return GetScopeInfo(GetScopeFor(class_id)).name.str();
  }

  // Returns the IR name to use for an instruction, when referenced from a given
  // scope.
  auto GetNameFor(ScopeIndex scope_idx, InstId inst_id) -> std::string {
    if (!inst_id.is_valid()) {
      return "invalid";
    }

    // Check for a builtin.
    if (inst_id.index < BuiltinKind::ValidCount) {
      return BuiltinKind::FromInt(inst_id.index).label().str();
    }

    auto& [inst_scope, inst_name] = insts[inst_id.index];
    if (!inst_name) {
      // This should not happen in valid IR.
      return "<unexpected instref " + llvm::itostr(inst_id.index) + ">";
    }
    if (inst_scope == scope_idx) {
      return inst_name.str().str();
    }
    return (GetScopeInfo(inst_scope).name.str() + "." + inst_name.str()).str();
  }

  // Returns the IR name to use for a label, when referenced from a given scope.
  auto GetLabelFor(ScopeIndex scope_idx, InstBlockId block_id) -> std::string {
    if (!block_id.is_valid()) {
      return "!invalid";
    }

    auto& [label_scope, label_name] = labels[block_id.index];
    if (!label_name) {
      // This should not happen in valid IR.
      return "<unexpected instblockref " + llvm::itostr(block_id.index) + ">";
    }
    if (label_scope == scope_idx) {
      return label_name.str().str();
    }
    return (GetScopeInfo(label_scope).name.str() + "." + label_name.str())
        .str();
  }

 private:
  // A space in which unique names can be allocated.
  struct Namespace {
    // A result of a name lookup.
    struct NameResult;

    // A name in a namespace, which might be redirected to refer to another name
    // for disambiguation purposes.
    class Name {
     public:
      Name() : value_(nullptr) {}
      explicit Name(llvm::StringMapIterator<NameResult> it) : value_(&*it) {}

      explicit operator bool() const { return value_; }

      auto str() const -> llvm::StringRef {
        llvm::StringMapEntry<NameResult>* value = value_;
        CARBON_CHECK(value) << "cannot print a null name";
        while (value->second.ambiguous && value->second.fallback) {
          value = value->second.fallback.value_;
        }
        return value->first();
      }

      auto SetFallback(Name name) -> void { value_->second.fallback = name; }

      auto SetAmbiguous() -> void { value_->second.ambiguous = true; }

     private:
      llvm::StringMapEntry<NameResult>* value_ = nullptr;
    };

    struct NameResult {
      bool ambiguous = false;
      Name fallback = Name();
    };

    llvm::StringRef prefix;
    llvm::StringMap<NameResult> allocated = {};
    int unnamed_count = 0;

    auto AddNameUnchecked(llvm::StringRef name) -> Name {
      return Name(allocated.insert({name, NameResult()}).first);
    }

    auto AllocateName(const InstNamer& namer, Parse::Node node,
                      std::string name = "") -> Name {
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

      // All names start with the prefix.
      name.insert(0, prefix);

      // Use the given name if it's available and not just the prefix.
      if (name.size() > prefix.size()) {
        add_name();
      }

      // Append location information to try to disambiguate.
      if (node.is_valid()) {
        auto token = namer.parse_tree_.node_token(node);
        llvm::raw_string_ostream(name)
            << ".loc" << namer.tokenized_buffer_.GetLineNumber(token);
        add_name();

        llvm::raw_string_ostream(name)
            << "_" << namer.tokenized_buffer_.GetColumnNumber(token);
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
  };

  // A named scope that contains named entities.
  struct Scope {
    Namespace::Name name;
    Namespace insts = {.prefix = "%"};
    Namespace labels = {.prefix = "!"};
  };

  auto GetScopeInfo(ScopeIndex scope_idx) -> Scope& {
    return scopes[static_cast<int>(scope_idx)];
  }

  auto AddBlockLabel(ScopeIndex scope_idx, InstBlockId block_id,
                     std::string name = "",
                     Parse::Node parse_node = Parse::Node::Invalid) -> void {
    if (!block_id.is_valid() || labels[block_id.index].second) {
      return;
    }

    if (parse_node == Parse::Node::Invalid) {
      if (const auto& block = sem_ir_.inst_blocks().Get(block_id);
          !block.empty()) {
        parse_node = sem_ir_.insts().Get(block.front()).parse_node();
      }
    }

    labels[block_id.index] = {scope_idx,
                              GetScopeInfo(scope_idx).labels.AllocateName(
                                  *this, parse_node, std::move(name))};
  }

  // Finds and adds a suitable block label for the given SemIR instruction that
  // represents some kind of branch.
  auto AddBlockLabel(ScopeIndex scope_idx, InstBlockId block_id, Inst inst)
      -> void {
    llvm::StringRef name;
    switch (parse_tree_.node_kind(inst.parse_node())) {
      case Parse::NodeKind::IfExpressionIf:
        switch (inst.kind()) {
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
        switch (inst.kind()) {
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

      case Parse::NodeKind::ShortCircuitOperand: {
        bool is_rhs = inst.Is<BranchIf>();
        bool is_and = tokenized_buffer_.GetKind(parse_tree_.node_token(
                          inst.parse_node())) == Lex::TokenKind::And;
        name = is_and ? (is_rhs ? "and.rhs" : "and.result")
                      : (is_rhs ? "or.rhs" : "or.result");
        break;
      }

      case Parse::NodeKind::WhileConditionStart:
        name = "while.cond";
        break;

      case Parse::NodeKind::WhileCondition:
        switch (inst.kind()) {
          case InstKind::BranchIf:
            name = "while.body";
            break;
          case InstKind::Branch:
            name = "while.done";
            break;
          default:
            break;
        }
        break;

      default:
        break;
    }

    AddBlockLabel(scope_idx, block_id, name.str(), inst.parse_node());
  }

  auto CollectNamesInBlock(ScopeIndex scope_idx, InstBlockId block_id) -> void {
    if (!block_id.is_valid()) {
      return;
    }

    Scope& scope = GetScopeInfo(scope_idx);

    // Use bound names where available. Otherwise, assign a backup name.
    for (auto inst_id : sem_ir_.inst_blocks().Get(block_id)) {
      if (!inst_id.is_valid()) {
        continue;
      }

      auto inst = sem_ir_.insts().Get(inst_id);
      auto add_inst_name = [&](std::string name) {
        insts[inst_id.index] = {scope_idx, scope.insts.AllocateName(
                                               *this, inst.parse_node(), name)};
      };
      auto add_inst_name_id = [&](IdentifierId name_id,
                                  llvm::StringRef suffix = "") {
        if (name_id.is_valid()) {
          add_inst_name(
              (sem_ir_.identifiers().Get(name_id).str() + suffix).str());
        } else {
          add_inst_name(suffix.str());
        }
      };

      switch (inst.kind()) {
        case Branch::Kind: {
          AddBlockLabel(scope_idx, inst.As<Branch>().target_id, inst);
          break;
        }
        case BranchIf::Kind: {
          AddBlockLabel(scope_idx, inst.As<BranchIf>().target_id, inst);
          break;
        }
        case BranchWithArg::Kind: {
          AddBlockLabel(scope_idx, inst.As<BranchWithArg>().target_id, inst);
          break;
        }
        case SpliceBlock::Kind: {
          CollectNamesInBlock(scope_idx, inst.As<SpliceBlock>().block_id);
          break;
        }
        case BindName::Kind: {
          add_inst_name_id(inst.As<BindName>().name_id);
          continue;
        }
        case FunctionDeclaration::Kind: {
          add_inst_name_id(sem_ir_.functions()
                               .Get(inst.As<FunctionDeclaration>().function_id)
                               .name_id);
          continue;
        }
        case ClassType::Kind: {
          add_inst_name_id(
              sem_ir_.classes().Get(inst.As<ClassType>().class_id).name_id);
          continue;
        }
        case NameReference::Kind: {
          add_inst_name_id(inst.As<NameReference>().name_id, ".ref");
          continue;
        }
        case Parameter::Kind: {
          add_inst_name_id(inst.As<Parameter>().name_id);
          continue;
        }
        case SelfParameter::Kind: {
          add_inst_name(inst.As<SelfParameter>().is_addr_self.index
                            ? "self.addr"
                            : "self");
          continue;
        }
        case VarStorage::Kind: {
          add_inst_name_id(inst.As<VarStorage>().name_id, ".var");
          continue;
        }
        default: {
          break;
        }
      }

      // Sequentially number all remaining values.
      if (inst.kind().value_kind() != InstValueKind::None) {
        add_inst_name("");
      }
    }
  }

  const Lex::TokenizedBuffer& tokenized_buffer_;
  const Parse::Tree& parse_tree_;
  const File& sem_ir_;

  Namespace globals = {.prefix = "@"};
  std::vector<std::pair<ScopeIndex, Namespace::Name>> insts;
  std::vector<std::pair<ScopeIndex, Namespace::Name>> labels;
  std::vector<Scope> scopes;
};
}  // namespace

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  explicit Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& sem_ir,
                     llvm::raw_ostream& out)
      : sem_ir_(sem_ir),
        out_(out),
        inst_namer_(tokenized_buffer, parse_tree, sem_ir) {}

  auto Format() -> void {
    out_ << "file \"" << sem_ir_.filename() << "\" {\n";
    // TODO: Include information from the `package` declaration, once we
    // fully support it.
    // TODO: Handle the case where there are multiple top-level instruction
    // blocks. For example, there may be branching in the initializer of a
    // global or a type expression.
    if (auto block_id = sem_ir_.top_inst_block_id(); block_id.is_valid()) {
      llvm::SaveAndRestore file_scope(scope_, InstNamer::ScopeIndex::File);
      FormatCodeBlock(block_id);
    }
    out_ << "}\n";

    for (int i : llvm::seq(sem_ir_.classes().size())) {
      FormatClass(ClassId(i));
    }

    for (int i : llvm::seq(sem_ir_.functions().size())) {
      FormatFunction(FunctionId(i));
    }
  }

  auto FormatClass(ClassId id) -> void {
    const Class& class_info = sem_ir_.classes().Get(id);

    out_ << "\nclass ";
    FormatClassName(id);

    llvm::SaveAndRestore class_scope(scope_, inst_namer_.GetScopeFor(id));

    if (class_info.scope_id.is_valid()) {
      out_ << " {\n";
      FormatCodeBlock(class_info.body_block_id);
      out_ << "\n!members:";
      FormatNameScope(class_info.scope_id, "", "\n  .");
      out_ << "\n}\n";
    } else {
      out_ << ";\n";
    }
  }

  auto FormatFunction(FunctionId id) -> void {
    const Function& fn = sem_ir_.functions().Get(id);

    out_ << "\nfn ";
    FormatFunctionName(id);

    llvm::SaveAndRestore function_scope(scope_, inst_namer_.GetScopeFor(id));

    if (fn.implicit_param_refs_id != SemIR::InstBlockId::Empty) {
      out_ << "[";
      FormatParameterList(fn.implicit_param_refs_id);
      out_ << "]";
    }

    out_ << "(";
    FormatParameterList(fn.param_refs_id);
    out_ << ")";

    if (fn.return_type_id.is_valid()) {
      out_ << " -> ";
      if (fn.return_slot_id.is_valid()) {
        FormatInstName(fn.return_slot_id);
        out_ << ": ";
      }
      FormatType(fn.return_type_id);
    }

    if (!fn.body_block_ids.empty()) {
      out_ << " {";

      for (auto block_id : fn.body_block_ids) {
        out_ << "\n";

        FormatLabel(block_id);
        out_ << ":\n";

        FormatCodeBlock(block_id);
      }

      out_ << "}\n";
    } else {
      out_ << ";\n";
    }
  }

  auto FormatParameterList(InstBlockId param_refs_id) -> void {
    llvm::ListSeparator sep;
    for (const InstId param_id : sem_ir_.inst_blocks().Get(param_refs_id)) {
      out_ << sep;
      if (!param_id.is_valid()) {
        out_ << "invalid";
        continue;
      }
      FormatInstName(param_id);
      out_ << ": ";
      FormatType(sem_ir_.insts().Get(param_id).type_id());
    }
  }

  auto FormatCodeBlock(InstBlockId block_id) -> void {
    if (!block_id.is_valid()) {
      return;
    }

    for (const InstId inst_id : sem_ir_.inst_blocks().Get(block_id)) {
      FormatInstruction(inst_id);
    }
  }

  auto FormatNameScope(NameScopeId id, llvm::StringRef separator,
                       llvm::StringRef prefix) -> void {
    // Name scopes aren't kept in any particular order. Sort the entries before
    // we print them for stability and consistency.
    llvm::SmallVector<std::pair<InstId, IdentifierId>> entries;
    for (auto [name_id, inst_id] : sem_ir_.name_scopes().Get(id)) {
      entries.push_back({inst_id, name_id});
    }
    llvm::sort(entries,
               [](auto a, auto b) { return a.first.index < b.first.index; });

    llvm::ListSeparator sep(separator);
    for (auto [inst_id, name_id] : entries) {
      out_ << sep << prefix;
      FormatString(name_id);
      out_ << " = ";
      FormatInstName(inst_id);
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
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (inst.kind()) {
#define CARBON_SEM_IR_INST_KIND(InstT)            \
  case InstT::Kind:                               \
    FormatInstruction(inst_id, inst.As<InstT>()); \
    break;
#include "toolchain/sem_ir/inst_kind.def"
    }
  }

  auto Indent() -> void { out_.indent(indent_); }

  template <typename InstT>
  auto FormatInstruction(InstId inst_id, InstT inst) -> void {
    Indent();
    FormatInstructionLHS(inst_id, inst);
    out_ << InstT::Kind.ir_name();
    FormatInstructionRHS(inst);
    out_ << "\n";
  }

  auto FormatInstructionLHS(InstId inst_id, Inst inst) -> void {
    switch (inst.kind().value_kind()) {
      case InstValueKind::Typed:
        FormatInstName(inst_id);
        out_ << ": ";
        switch (GetExpressionCategory(sem_ir_, inst_id)) {
          case ExpressionCategory::NotExpression:
          case ExpressionCategory::Error:
          case ExpressionCategory::Value:
          case ExpressionCategory::Mixed:
            break;
          case ExpressionCategory::DurableReference:
          case ExpressionCategory::EphemeralReference:
            out_ << "ref ";
            break;
          case ExpressionCategory::Initializing:
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

  template <typename InstT>
  auto FormatInstructionRHS(InstT inst) -> void {
    // By default, an instruction has a comma-separated argument list.
    using Info = TypedInstArgsInfo<InstT>;
    if constexpr (Info::NumArgs == 2) {
      FormatArgs(Info::template Get<0>(inst), Info::template Get<1>(inst));
    } else if constexpr (Info::NumArgs == 1) {
      FormatArgs(Info::template Get<0>(inst));
    } else {
      FormatArgs();
    }
  }

  auto FormatInstructionRHS(BlockArg inst) -> void {
    out_ << " ";
    FormatLabel(inst.block_id);
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

  auto FormatInstructionRHS(ArrayInit inst) -> void {
    out_ << " ";
    FormatArg(inst.tuple_id);

    llvm::ArrayRef<InstId> inits_and_return_slot =
        sem_ir_.inst_blocks().Get(inst.inits_and_return_slot_id);
    auto inits = inits_and_return_slot.drop_back(1);
    auto return_slot_id = inits_and_return_slot.back();

    out_ << ", (";
    llvm::ListSeparator sep;
    for (auto inst_id : inits) {
      out_ << sep;
      FormatArg(inst_id);
    }
    out_ << ')';
    FormatReturnSlot(return_slot_id);
  }

  auto FormatInstructionRHS(Call inst) -> void {
    out_ << " ";
    FormatArg(inst.callee_id);

    if (!inst.args_id.is_valid()) {
      out_ << "(<invalid>)";
      return;
    }

    llvm::ArrayRef<InstId> args = sem_ir_.inst_blocks().Get(inst.args_id);

    bool has_return_slot =
        GetInitializingRepresentation(sem_ir_, inst.type_id).has_return_slot();
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

  auto FormatInstructionRHS(InitializeFrom inst) -> void {
    FormatArgs(inst.src_id);
    FormatReturnSlot(inst.dest_id);
  }

  auto FormatInstructionRHS(CrossReference inst) -> void {
    // TODO: Figure out a way to make this meaningful. We'll need some way to
    // name cross-reference IRs, perhaps by the instruction ID of the import?
    out_ << " " << inst.ir_id << "." << inst.inst_id;
  }

  auto FormatInstructionRHS(SpliceBlock inst) -> void {
    FormatArgs(inst.result_id);
    out_ << " {";
    if (!sem_ir_.inst_blocks().Get(inst.block_id).empty()) {
      out_ << "\n";
      indent_ += 2;
      FormatCodeBlock(inst.block_id);
      indent_ -= 2;
      Indent();
    }
    out_ << "}";
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
      FormatString(field.name_id);
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

  auto FormatArg(FunctionId id) -> void { FormatFunctionName(id); }

  auto FormatArg(ClassId id) -> void { FormatClassName(id); }

  auto FormatArg(IdentifierId id) -> void {
    out_ << '"';
    out_.write_escaped(sem_ir_.identifiers().Get(id), /*UseHexEscapes=*/true);
    out_ << '"';
  }

  auto FormatArg(IntegerId id) -> void {
    sem_ir_.integers().Get(id).print(out_, /*isSigned=*/false);
  }

  auto FormatArg(MemberIndex index) -> void { out_ << index; }

  auto FormatArg(NameScopeId id) -> void {
    out_ << '{';
    FormatNameScope(id, ", ", ".");
    out_ << '}';
  }

  auto FormatArg(InstId id) -> void { FormatInstName(id); }

  auto FormatArg(InstBlockId id) -> void {
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

  auto FormatArg(StringLiteralId id) -> void {
    out_ << '"';
    out_.write_escaped(sem_ir_.string_literals().Get(id),
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

  auto FormatInstName(InstId id) -> void {
    out_ << inst_namer_.GetNameFor(scope_, id);
  }

  auto FormatLabel(InstBlockId id) -> void {
    out_ << inst_namer_.GetLabelFor(scope_, id);
  }

  auto FormatString(IdentifierId id) -> void {
    out_ << sem_ir_.identifiers().Get(id);
  }

  auto FormatFunctionName(FunctionId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatClassName(ClassId id) -> void {
    out_ << inst_namer_.GetNameFor(id);
  }

  auto FormatType(TypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << sem_ir_.StringifyType(id, /*in_type_context=*/true);
    }
  }

 private:
  const File& sem_ir_;
  llvm::raw_ostream& out_;
  InstNamer inst_namer_;
  InstNamer::ScopeIndex scope_ = InstNamer::ScopeIndex::None;
  bool in_terminator_sequence_ = false;
  int indent_ = 2;
};

auto FormatFile(const Lex::TokenizedBuffer& tokenized_buffer,
                const Parse::Tree& parse_tree, const File& sem_ir,
                llvm::raw_ostream& out) -> void {
  Formatter(tokenized_buffer, parse_tree, sem_ir, out).Format();
}

}  // namespace Carbon::SemIR
