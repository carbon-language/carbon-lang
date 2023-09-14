// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/formatter.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"

namespace Carbon::SemIR {

namespace {
// Assigns names to nodes, blocks, and scopes in the Semantics IR.
//
// TODOs / future work ideas:
// - Add a documentation file for the textual format and link to the
//   naming section here.
// - Consider representing literals as just `literal` in the IR and using the
//   type to distinguish.
class NodeNamer {
 public:
  // int32_t matches the input value size.
  // NOLINTNEXTLINE(performance-enum-size)
  enum class ScopeIndex : int32_t {
    None = -1,
    Package = 0,
  };
  static_assert(sizeof(ScopeIndex) == sizeof(FunctionId));

  NodeNamer(const Lex::TokenizedBuffer& tokenized_buffer,
            const Parse::Tree& parse_tree, const File& semantics_ir)
      : tokenized_buffer_(tokenized_buffer),
        parse_tree_(parse_tree),
        semantics_ir_(semantics_ir) {
    nodes.resize(semantics_ir.nodes_size());
    labels.resize(semantics_ir.node_blocks_size());
    scopes.resize(1 + semantics_ir.functions_size());

    // Build the package scope.
    GetScopeInfo(ScopeIndex::Package).name =
        globals.AddNameUnchecked("package");
    CollectNamesInBlock(ScopeIndex::Package, semantics_ir.top_node_block_id());

    // Build each function scope.
    for (int i : llvm::seq(semantics_ir.functions_size())) {
      auto fn_id = FunctionId(i);
      auto fn_scope = GetScopeFor(fn_id);
      const auto& fn = semantics_ir.GetFunction(fn_id);
      // TODO: Provide a location for the function for use as a
      // disambiguator.
      auto fn_loc = Parse::Node::Invalid;
      GetScopeInfo(fn_scope).name = globals.AllocateName(
          *this, fn_loc,
          fn.name_id.is_valid() ? semantics_ir.GetString(fn.name_id).str()
                                : "");
      CollectNamesInBlock(fn_scope, fn.param_refs_id);
      if (fn.return_slot_id.is_valid()) {
        nodes[fn.return_slot_id.index] = {
            fn_scope,
            GetScopeInfo(fn_scope).nodes.AllocateName(
                *this, semantics_ir.GetNode(fn.return_slot_id).parse_node(),
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
  }

  // Returns the scope index corresponding to a function.
  auto GetScopeFor(FunctionId fn_id) -> ScopeIndex {
    return static_cast<ScopeIndex>(fn_id.index + 1);
  }

  // Returns the IR name to use for a function.
  auto GetNameFor(FunctionId fn_id) -> llvm::StringRef {
    if (!fn_id.is_valid()) {
      return "invalid";
    }
    return GetScopeInfo(GetScopeFor(fn_id)).name.str();
  }

  // Returns the IR name to use for a node, when referenced from a given scope.
  auto GetNameFor(ScopeIndex scope_idx, NodeId node_id) -> std::string {
    if (!node_id.is_valid()) {
      return "invalid";
    }

    // Check for a builtin.
    if (node_id.index < BuiltinKind::ValidCount) {
      return BuiltinKind::FromInt(node_id.index).label().str();
    }

    auto& [node_scope, node_name] = nodes[node_id.index];
    if (!node_name) {
      // This should not happen in valid IR.
      return "<unexpected noderef " + llvm::itostr(node_id.index) + ">";
    }
    if (node_scope == scope_idx) {
      return node_name.str().str();
    }
    return (GetScopeInfo(node_scope).name.str() + "." + node_name.str()).str();
  }

  // Returns the IR name to use for a label, when referenced from a given scope.
  auto GetLabelFor(ScopeIndex scope_idx, NodeBlockId block_id) -> std::string {
    if (!block_id.is_valid()) {
      return "!invalid";
    }

    auto& [label_scope, label_name] = labels[block_id.index];
    if (!label_name) {
      // This should not happen in valid IR.
      return "<unexpected nodeblockref " + llvm::itostr(block_id.index) + ">";
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
      llvm::StringMapEntry<NameResult>* value_;
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

    auto AllocateName(const NodeNamer& namer, Parse::Node node,
                      std::string name = "") -> Name {
      // The best (shortest) name for this node so far, and the current name
      // for it.
      Name best;
      Name current;

      // Add `name` as a name for this entity.
      auto add_name = [&](bool mark_ambiguous = true) {
        auto [it, added] = allocated.insert({name, NameResult()});
        Name new_name = Name(it);

        if (!added) {
          if (mark_ambiguous) {
            // This name was allocated for a different node. Mark it as
            // ambiguous and keep looking for a name for this node.
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
    Namespace nodes = {.prefix = "%"};
    Namespace labels = {.prefix = "!"};
  };

  auto GetScopeInfo(ScopeIndex scope_idx) -> Scope& {
    return scopes[static_cast<int>(scope_idx)];
  }

  auto AddBlockLabel(ScopeIndex scope_idx, NodeBlockId block_id,
                     std::string name = "",
                     Parse::Node parse_node = Parse::Node::Invalid) -> void {
    if (!block_id.is_valid() || labels[block_id.index].second) {
      return;
    }

    if (parse_node == Parse::Node::Invalid) {
      if (const auto& block = semantics_ir_.GetNodeBlock(block_id);
          !block.empty()) {
        parse_node = semantics_ir_.GetNode(block.front()).parse_node();
      }
    }

    labels[block_id.index] = {scope_idx,
                              GetScopeInfo(scope_idx).labels.AllocateName(
                                  *this, parse_node, std::move(name))};
  }

  // Finds and adds a suitable block label for the given semantics node that
  // represents some kind of branch.
  auto AddBlockLabel(ScopeIndex scope_idx, NodeBlockId block_id, Node node)
      -> void {
    llvm::StringRef name;
    switch (parse_tree_.node_kind(node.parse_node())) {
      case Parse::NodeKind::IfExpressionIf:
        switch (node.kind()) {
          case NodeKind::BranchIf:
            name = "if.expr.then";
            break;
          case NodeKind::Branch:
            name = "if.expr.else";
            break;
          case NodeKind::BranchWithArg:
            name = "if.expr.result";
            break;
          default:
            break;
        }
        break;

      case Parse::NodeKind::IfCondition:
        switch (node.kind()) {
          case NodeKind::BranchIf:
            name = "if.then";
            break;
          case NodeKind::Branch:
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
        bool is_rhs = node.kind() == NodeKind::BranchIf;
        bool is_and = tokenized_buffer_.GetKind(parse_tree_.node_token(
                          node.parse_node())) == Lex::TokenKind::And;
        name = is_and ? (is_rhs ? "and.rhs" : "and.result")
                      : (is_rhs ? "or.rhs" : "or.result");
        break;
      }

      default:
        break;
    }

    AddBlockLabel(scope_idx, block_id, name.str(), node.parse_node());
  }

  auto CollectNamesInBlock(ScopeIndex scope_idx, NodeBlockId block_id) -> void {
    if (!block_id.is_valid()) {
      return;
    }

    Scope& scope = GetScopeInfo(scope_idx);

    // Use bound names where available. Otherwise, assign a backup name.
    for (auto node_id : semantics_ir_.GetNodeBlock(block_id)) {
      if (!node_id.is_valid()) {
        continue;
      }
      auto node = semantics_ir_.GetNode(node_id);
      switch (node.kind()) {
        case NodeKind::Branch: {
          auto dest_id = node.GetAsBranch();
          AddBlockLabel(scope_idx, dest_id, node);
          break;
        }
        case NodeKind::BranchIf: {
          auto [dest_id, cond_id] = node.GetAsBranchIf();
          AddBlockLabel(scope_idx, dest_id, node);
          break;
        }
        case NodeKind::BranchWithArg: {
          auto [dest_id, arg_id] = node.GetAsBranchWithArg();
          AddBlockLabel(scope_idx, dest_id, node);
          break;
        }
        case NodeKind::Parameter: {
          auto name_id = node.GetAsParameter();
          nodes[node_id.index] = {
              scope_idx,
              scope.nodes.AllocateName(*this, node.parse_node(),
                                       semantics_ir_.GetString(name_id).str())};
          break;
        }
        case NodeKind::VarStorage: {
          // TODO: Eventually this name will be optional, and we'll want to
          // provide something like `var` as a default. However, that's not
          // possible right now so cannot be tested.
          auto name_id = node.GetAsVarStorage();
          nodes[node_id.index] = {
              scope_idx,
              scope.nodes.AllocateName(*this, node.parse_node(),
                                       semantics_ir_.GetString(name_id).str())};
          break;
        }
        default: {
          // Sequentially number all remaining values.
          if (node.kind().value_kind() != NodeValueKind::None) {
            nodes[node_id.index] = {
                scope_idx, scope.nodes.AllocateName(*this, node.parse_node())};
          }
          break;
        }
      }
    }
  }

  const Lex::TokenizedBuffer& tokenized_buffer_;
  const Parse::Tree& parse_tree_;
  const File& semantics_ir_;

  Namespace globals = {.prefix = "@"};
  std::vector<std::pair<ScopeIndex, Namespace::Name>> nodes;
  std::vector<std::pair<ScopeIndex, Namespace::Name>> labels;
  std::vector<Scope> scopes;
};
}  // namespace

// Formatter for printing textual Semantics IR.
class Formatter {
 public:
  explicit Formatter(const Lex::TokenizedBuffer& tokenized_buffer,
                     const Parse::Tree& parse_tree, const File& semantics_ir,
                     llvm::raw_ostream& out)
      : semantics_ir_(semantics_ir),
        out_(out),
        node_namer_(tokenized_buffer, parse_tree, semantics_ir) {}

  auto Format() -> void {
    out_ << "file \"" << semantics_ir_.filename() << "\" {\n";
    // TODO: Include information from the package declaration, once we
    // fully support it.
    // TODO: Handle the case where there are multiple top-level node blocks.
    // For example, there may be branching in the initializer of a global or a
    // type expression.
    if (auto block_id = semantics_ir_.top_node_block_id();
        block_id.is_valid()) {
      llvm::SaveAndRestore package_scope(scope_,
                                         NodeNamer::ScopeIndex::Package);
      FormatCodeBlock(block_id);
    }
    out_ << "}\n";

    for (int i : llvm::seq(semantics_ir_.functions_size())) {
      FormatFunction(FunctionId(i));
    }
  }

  auto FormatFunction(FunctionId id) -> void {
    const Function& fn = semantics_ir_.GetFunction(id);

    out_ << "\nfn ";
    FormatFunctionName(id);
    out_ << "(";

    llvm::SaveAndRestore function_scope(scope_, node_namer_.GetScopeFor(id));

    llvm::ListSeparator sep;
    for (const NodeId param_id : semantics_ir_.GetNodeBlock(fn.param_refs_id)) {
      out_ << sep;
      if (!param_id.is_valid()) {
        out_ << "invalid";
        continue;
      }
      FormatNodeName(param_id);
      out_ << ": ";
      FormatType(semantics_ir_.GetNode(param_id).type_id());
    }
    out_ << ")";
    if (fn.return_type_id.is_valid()) {
      out_ << " -> ";
      if (fn.return_slot_id.is_valid()) {
        FormatNodeName(fn.return_slot_id);
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

  auto FormatCodeBlock(NodeBlockId block_id) -> void {
    if (!block_id.is_valid()) {
      return;
    }

    for (const NodeId node_id : semantics_ir_.GetNodeBlock(block_id)) {
      FormatInstruction(node_id);
    }
  }

  auto FormatInstruction(NodeId node_id) -> void {
    if (!node_id.is_valid()) {
      out_ << "  " << NodeKind::Invalid.ir_name() << "\n";
      return;
    }

    FormatInstruction(node_id, semantics_ir_.GetNode(node_id));
  }

  auto FormatInstruction(NodeId node_id, Node node) -> void {
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)          \
  case NodeKind::Name:                            \
    FormatInstruction<Node::Name>(node_id, node); \
    break;
#include "toolchain/sem_ir/node_kind.def"
    }
  }

  template <typename Kind>
  auto FormatInstruction(NodeId node_id, Node node) -> void {
    out_ << "  ";
    FormatInstructionLHS(node_id, node);
    out_ << node.kind().ir_name();
    FormatInstructionRHS<Kind>(node);
    out_ << "\n";
  }

  auto FormatInstructionLHS(NodeId node_id, Node node) -> void {
    switch (node.kind().value_kind()) {
      case NodeValueKind::Typed:
        FormatNodeName(node_id);
        out_ << ": ";
        switch (GetExpressionCategory(semantics_ir_, node_id)) {
          case ExpressionCategory::NotExpression:
          case ExpressionCategory::Value:
            break;
          case ExpressionCategory::DurableReference:
          case ExpressionCategory::EphemeralReference:
            out_ << "ref ";
            break;
          case ExpressionCategory::Initializing:
            out_ << "init ";
            break;
        }
        FormatType(node.type_id());
        out_ << " = ";
        break;
      case NodeValueKind::Untyped:
        FormatNodeName(node_id);
        out_ << " = ";
        break;
      case NodeValueKind::None:
        break;
    }
  }

  template <typename Kind>
  auto FormatInstructionRHS(Node node) -> void {
    // By default, an instruction has a comma-separated argument list.
    FormatArgs(Kind::Get(node));
  }

  template <>
  auto FormatInstructionRHS<Node::BlockArg>(Node node) -> void {
    out_ << " ";
    FormatLabel(node.GetAsBlockArg());
  }

  template <>
  auto FormatInstruction<Node::BranchIf>(NodeId /*node_id*/, Node node)
      -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    auto [label_id, cond_id] = node.GetAsBranchIf();
    out_ << "if ";
    FormatNodeName(cond_id);
    out_ << " " << NodeKind::Branch.ir_name() << " ";
    FormatLabel(label_id);
    out_ << " else ";
    in_terminator_sequence = true;
  }

  template <>
  auto FormatInstruction<Node::BranchWithArg>(NodeId /*node_id*/, Node node)
      -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    auto [label_id, arg_id] = node.GetAsBranchWithArg();
    out_ << NodeKind::BranchWithArg.ir_name() << " ";
    FormatLabel(label_id);
    out_ << "(";
    FormatNodeName(arg_id);
    out_ << ")\n";
    in_terminator_sequence = false;
  }

  template <>
  auto FormatInstruction<Node::Branch>(NodeId /*node_id*/, Node node) -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    out_ << NodeKind::Branch.ir_name() << " ";
    FormatLabel(node.GetAsBranch());
    out_ << "\n";
    in_terminator_sequence = false;
  }

  template <>
  auto FormatInstructionRHS<Node::ArrayInit>(Node node) -> void {
    out_ << " ";
    auto [src_id, refs_id] = node.GetAsArrayInit();
    FormatArg(src_id);

    llvm::ArrayRef<NodeId> refs = semantics_ir_.GetNodeBlock(refs_id);
    auto inits = refs.drop_back(1);
    auto return_slot_id = refs.back();

    out_ << ", (";
    llvm::ListSeparator sep;
    for (auto node_id : inits) {
      out_ << sep;
      FormatArg(node_id);
    }
    out_ << ')';
    FormatReturnSlot(return_slot_id);
  }

  template <>
  auto FormatInstructionRHS<Node::Call>(Node node) -> void {
    out_ << " ";
    auto [args_id, callee_id] = node.GetAsCall();
    FormatArg(callee_id);

    llvm::ArrayRef<NodeId> args = semantics_ir_.GetNodeBlock(args_id);

    bool has_return_slot =
        semantics_ir_.GetFunction(callee_id).return_slot_id.is_valid();
    NodeId return_slot_id = NodeId::Invalid;
    if (has_return_slot) {
      return_slot_id = args.back();
      args = args.drop_back();
    }

    llvm::ListSeparator sep;
    out_ << '(';
    for (auto node_id : args) {
      out_ << sep;
      FormatArg(node_id);
    }
    out_ << ')';

    if (has_return_slot) {
      FormatReturnSlot(return_slot_id);
    }
  }

  template <>
  auto FormatInstructionRHS<Node::InitializeFrom>(Node node) -> void {
    auto [src_id, dest_id] = node.GetAsInitializeFrom();
    FormatArgs(src_id);
    FormatReturnSlot(dest_id);
  }

  template <>
  auto FormatInstructionRHS<Node::CrossReference>(Node node) -> void {
    // TODO: Figure out a way to make this meaningful. We'll need some way to
    // name cross-reference IRs, perhaps by the node ID of the import?
    auto [xref_id, node_id] = node.GetAsCrossReference();
    out_ << " " << xref_id << "." << node_id;
  }

  // StructTypeFields are formatted as part of their StructType.
  template <>
  auto FormatInstruction<Node::StructTypeField>(NodeId /*node_id*/,
                                                Node /*node*/) -> void {}

  template <>
  auto FormatInstructionRHS<Node::StructType>(Node node) -> void {
    out_ << " {";
    llvm::ListSeparator sep;
    for (auto field_id : semantics_ir_.GetNodeBlock(node.GetAsStructType())) {
      out_ << sep << ".";
      auto [field_name_id, field_type_id] =
          semantics_ir_.GetNode(field_id).GetAsStructTypeField();
      FormatString(field_name_id);
      out_ << ": ";
      FormatType(field_type_id);
    }
    out_ << "}";
  }

  auto FormatArgs(Node::NoArgs /*unused*/) -> void {}

  template <typename Arg1>
  auto FormatArgs(Arg1 arg) -> void {
    out_ << ' ';
    FormatArg(arg);
  }

  template <typename Arg1, typename Arg2>
  auto FormatArgs(std::pair<Arg1, Arg2> args) -> void {
    out_ << ' ';
    FormatArg(args.first);
    out_ << ",";
    FormatArgs(args.second);
  }

  auto FormatArg(BoolValue v) -> void { out_ << v; }

  auto FormatArg(BuiltinKind kind) -> void { out_ << kind.label(); }

  auto FormatArg(FunctionId id) -> void { FormatFunctionName(id); }

  auto FormatArg(IntegerLiteralId id) -> void {
    semantics_ir_.GetIntegerLiteral(id).print(out_, /*isSigned=*/false);
  }

  auto FormatArg(MemberIndex index) -> void { out_ << index; }

  // TODO: Should we be printing scopes inline, or should we have a separate
  // step to print them like we do for functions?
  auto FormatArg(NameScopeId id) -> void {
    // Name scopes aren't kept in any particular order. Sort the entries before
    // we print them for stability and consistency.
    std::vector<std::pair<NodeId, StringId>> entries;
    for (auto [name_id, node_id] : semantics_ir_.GetNameScope(id)) {
      entries.push_back({node_id, name_id});
    }
    llvm::sort(entries,
               [](auto a, auto b) { return a.first.index < b.first.index; });

    out_ << '{';
    llvm::ListSeparator sep;
    for (auto [node_id, name_id] : entries) {
      out_ << sep << ".";
      FormatString(name_id);
      out_ << " = ";
      FormatNodeName(node_id);
    }
    out_ << '}';
  }

  auto FormatArg(NodeId id) -> void { FormatNodeName(id); }

  auto FormatArg(NodeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto node_id : semantics_ir_.GetNodeBlock(id)) {
      out_ << sep;
      FormatArg(node_id);
    }
    out_ << ')';
  }

  auto FormatArg(RealLiteralId id) -> void {
    // TODO: Format with a `.` when the exponent is near zero.
    const auto& real = semantics_ir_.GetRealLiteral(id);
    real.mantissa.print(out_, /*isSigned=*/false);
    out_ << (real.is_decimal ? 'e' : 'p') << real.exponent;
  }

  auto FormatArg(StringId id) -> void {
    out_ << '"';
    out_.write_escaped(semantics_ir_.GetString(id), /*UseHexEscapes=*/true);
    out_ << '"';
  }

  auto FormatArg(TypeId id) -> void { FormatType(id); }

  auto FormatArg(TypeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto type_id : semantics_ir_.GetTypeBlock(id)) {
      out_ << sep;
      FormatArg(type_id);
    }
    out_ << ')';
  }

  auto FormatReturnSlot(NodeId dest_id) -> void {
    out_ << " to ";
    FormatArg(dest_id);
  }

  auto FormatNodeName(NodeId id) -> void {
    out_ << node_namer_.GetNameFor(scope_, id);
  }

  auto FormatLabel(NodeBlockId id) -> void {
    out_ << node_namer_.GetLabelFor(scope_, id);
  }

  auto FormatString(StringId id) -> void {
    out_ << semantics_ir_.GetString(id);
  }

  auto FormatFunctionName(FunctionId id) -> void {
    out_ << node_namer_.GetNameFor(id);
  }

  auto FormatType(TypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << semantics_ir_.StringifyType(id, /*in_type_context=*/true);
    }
  }

 private:
  const File& semantics_ir_;
  llvm::raw_ostream& out_;
  NodeNamer node_namer_;
  NodeNamer::ScopeIndex scope_ = NodeNamer::ScopeIndex::None;
  bool in_terminator_sequence = false;
};

auto FormatFile(const Lex::TokenizedBuffer& tokenized_buffer,
                const Parse::Tree& parse_tree, const File& semantics_ir,
                llvm::raw_ostream& out) -> void {
  Formatter(tokenized_buffer, parse_tree, semantics_ir, out).Format();
}

}  // namespace Carbon::SemIR
