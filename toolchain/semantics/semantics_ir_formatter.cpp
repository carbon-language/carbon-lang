// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_formatter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

namespace {
// Assigns names to nodes, blocks, and scopes in the Semantics IR.
//
// TODOs / future work ideas:
// - Add a documentation file for the textual format and link to the
//   naming section here.
// - Consider harmonizing `FooLiteral` vs. `foo_value` node names vs.
//   IR names.
//   - Also consider representing these as just `value` or `literal` 
//     in the IR and using the type to distinguish.
// - Add block names based on the control flow construct names (`for`,
//   `if`, `then`, `else`, ...). Either base this on the semantics or
//   just on the keywords used to build that control flow -- little
//   or no harm to getting it wrong. Tools will also want to be able
//   to query the construct that resulted in control flow.
class NodeNamer {
 public:
  enum class ScopeIndex : int {
    None = -1,
    Package = 0,
  };

  NodeNamer(const TokenizedBuffer& tokenized_buffer,
            const ParseTree& parse_tree, const SemanticsIR& semantics_ir)
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
    for (int i = 0; i != semantics_ir.functions_size(); ++i) {
      auto fn_id = SemanticsFunctionId(i);
      auto fn_scope = GetScopeFor(fn_id);
      const auto& fn = semantics_ir.GetFunction(fn_id);
      GetScopeInfo(fn_scope).name = globals.AllocateName(
          *this,
          // TODO: Provide a location for the function for use as a
          // disambiguator.
          ParseTree::Node::Invalid,
          fn.name_id.is_valid() ? semantics_ir.GetString(fn.name_id).str()
                                : "");
      CollectNamesInBlock(fn_scope, fn.param_refs_id);
      for (auto block_id : fn.body_block_ids) {
        AddBlockLabel(fn_scope, block_id,
                      block_id == fn.body_block_ids.front() ? "entry" : "");
        CollectNamesInBlock(fn_scope, block_id);
      }
    }
  }

  // Returns the scope index corresponding to a function.
  auto GetScopeFor(SemanticsFunctionId fn_id) -> ScopeIndex {
    return ScopeIndex(fn_id.index + 1);
  }

  // Returns the IR name to use for a function.
  auto GetNameFor(SemanticsFunctionId fn_id) -> llvm::StringRef {
    if (!fn_id.is_valid()) {
      return "invalid";
    }
    return GetScopeInfo(GetScopeFor(fn_id)).name;
  }

  // Returns the IR name to use for a node, when referenced from a given scope.
  auto GetNameFor(ScopeIndex scope_idx, SemanticsNodeId node_id)
      -> std::string {
    if (!node_id.is_valid()) {
      return "invalid";
    }

    // Check for a builtin.
    if (node_id.index < SemanticsBuiltinKind::ValidCount) {
      return SemanticsBuiltinKind::FromInt(node_id.index).label().str();
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
  auto GetLabelFor(ScopeIndex scope_idx, SemanticsNodeBlockId block_id)
      -> std::string {
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

      operator llvm::StringRef() const { return str(); }

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

    auto AllocateName(const NodeNamer& namer, ParseTree::Node node,
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
    return scopes[(int)scope_idx];
  }

  auto AddBlockLabel(ScopeIndex scope_idx, SemanticsNodeBlockId block_id,
                     std::string name = "") -> void {
    if (!block_id.is_valid()) {
      return;
    }

    ParseTree::Node parse_node = ParseTree::Node::Invalid;
    if (auto& block = semantics_ir_.GetNodeBlock(block_id); !block.empty()) {
      parse_node = semantics_ir_.GetNode(block.front()).parse_node();
    }

    labels[block_id.index] = {scope_idx,
                              GetScopeInfo(scope_idx).labels.AllocateName(
                                  *this, parse_node, std::move(name))};
  }

  auto CollectNamesInBlock(ScopeIndex scope_idx, SemanticsNodeBlockId block_id)
      -> void {
    if (!block_id.is_valid()) {
      return;
    }

    Scope& scope = GetScopeInfo(scope_idx);

    // Use bound names where available. The BindName node appears after the node
    // that it's giving a name to, so we need to do this before assigning
    // fallback names.
    for (auto node_id : semantics_ir_.GetNodeBlock(block_id)) {
      auto node = semantics_ir_.GetNode(node_id);
      if (node.kind() == SemanticsNodeKind::BindName) {
        auto [name_id, named_node_id] = node.GetAsBindName();
        nodes[named_node_id.index] = {
            scope_idx,
            scope.nodes.AllocateName(*this, node.parse_node(),
                                     semantics_ir_.GetString(name_id).str())};
      }
    }

    // Sequentially number all remaining values.
    for (auto node_id : semantics_ir_.GetNodeBlock(block_id)) {
      auto node = semantics_ir_.GetNode(node_id);
      if (node.kind() != SemanticsNodeKind::BindName &&
          node.kind().value_kind() != SemanticsNodeValueKind::None) {
        auto& name = nodes[node_id.index];
        if (!name.second) {
          name = {scope_idx,
                  scope.nodes.AllocateName(*this, node.parse_node())};
        }
      }
    }
  }

  const TokenizedBuffer& tokenized_buffer_;
  const ParseTree& parse_tree_;
  const SemanticsIR& semantics_ir_;

  Namespace globals = {.prefix = "@"};
  std::vector<std::pair<ScopeIndex, Namespace::Name>> nodes;
  std::vector<std::pair<ScopeIndex, Namespace::Name>> labels;
  std::vector<Scope> scopes;
};
}  // namespace

// Formatter for printing textual Semantics IR.
class SemanticsIRFormatter {
 public:
  explicit SemanticsIRFormatter(const TokenizedBuffer& tokenized_buffer,
                                const ParseTree& parse_tree,
                                const SemanticsIR& semantics_ir,
                                llvm::raw_ostream& out)
      : semantics_ir_(semantics_ir),
        out_(out),
        node_namer_(tokenized_buffer, parse_tree, semantics_ir) {}

  auto Format() -> void {
    // TODO: Include information from the package declaration, once we fully
    // support it.
    out_ << "package {\n";
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

    for (int i = 0; i != semantics_ir_.functions_size(); ++i) {
      FormatFunction(SemanticsFunctionId(i));
    }
  }

  auto FormatFunction(SemanticsFunctionId id) -> void {
    const SemanticsFunction& fn = semantics_ir_.GetFunction(id);

    out_ << "\nfn ";
    FormatFunctionName(id);
    out_ << "(";

    llvm::SaveAndRestore function_scope(scope_, node_namer_.GetScopeFor(id));

    llvm::ListSeparator sep;
    for (const SemanticsNodeId param_id :
         semantics_ir_.GetNodeBlock(fn.param_refs_id)) {
      out_ << sep;
      auto param = semantics_ir_.GetNode(param_id);
      auto [name_id, node_id] = param.GetAsBindName();
      FormatNodeName(node_id);
      out_ << ": ";
      FormatType(param.type_id());
    }
    out_ << ")";
    if (fn.return_type_id.is_valid()) {
      out_ << " -> ";
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

  auto FormatCodeBlock(SemanticsNodeBlockId block_id) -> void {
    if (!block_id.is_valid()) {
      return;
    }

    for (const SemanticsNodeId node_id : semantics_ir_.GetNodeBlock(block_id)) {
      FormatInstruction(node_id);
    }
  }

  auto FormatInstruction(SemanticsNodeId node_id) -> void {
    if (!node_id.is_valid()) {
      out_ << "  " << SemanticsNodeKind::Invalid.ir_name() << "\n";
      return;
    }

    FormatInstruction(node_id, semantics_ir_.GetNode(node_id));
  }

  auto FormatInstruction(SemanticsNodeId node_id, SemanticsNode node) -> void {
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)                   \
  case SemanticsNodeKind::Name:                            \
    FormatInstruction<SemanticsNode::Name>(node_id, node); \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
  }

  template <typename Kind>
  auto FormatInstruction(SemanticsNodeId node_id, SemanticsNode node) -> void {
    out_ << "  ";
    FormatInstructionLHS(node_id, node);
    out_ << node.kind().ir_name();
    FormatInstructionRHS<Kind>(node);
    out_ << "\n";
  }

  auto FormatInstructionLHS(SemanticsNodeId node_id, SemanticsNode node)
      -> void {
    switch (node.kind().value_kind()) {
      case SemanticsNodeValueKind::Typed:
        FormatNodeName(node_id);
        out_ << ": ";
        FormatType(node.type_id());
        out_ << " = ";
        break;
      case SemanticsNodeValueKind::Untyped:
        FormatNodeName(node_id);
        out_ << " = ";
        break;
      case SemanticsNodeValueKind::None:
        break;
    }
  }

  template <typename Kind>
  auto FormatInstructionRHS(SemanticsNode node) -> void {
    // By default, an instruction has a comma-separated argument list.
    FormatArgs(Kind::Get(node));
  }

  // BindName is handled by the NodeNamer and doesn't appear in the output.
  // These nodes are currently used simply to give a name to another node, and
  // are never referenced themselves.
  // TODO: Include BindName nodes in the output if we start referring to them.
  template <>
  auto FormatInstruction<SemanticsNode::BindName>(SemanticsNodeId,
                                                  SemanticsNode) -> void {}

  template <>
  auto FormatInstructionRHS<SemanticsNode::BlockArg>(SemanticsNode node)
      -> void {
    out_ << " ";
    FormatLabel(node.GetAsBlockArg());
  }

  template <>
  auto FormatInstruction<SemanticsNode::BranchIf>(SemanticsNodeId,
                                                  SemanticsNode node) -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    auto [label_id, cond_id] = node.GetAsBranchIf();
    out_ << "if ";
    FormatNodeName(cond_id);
    out_ << " " << SemanticsNodeKind::Branch.ir_name() << " ";
    FormatLabel(label_id);
    out_ << " else ";
    in_terminator_sequence = true;
  }

  template <>
  auto FormatInstruction<SemanticsNode::BranchWithArg>(SemanticsNodeId,
                                                       SemanticsNode node)
      -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    auto [label_id, arg_id] = node.GetAsBranchWithArg();
    out_ << SemanticsNodeKind::BranchWithArg.ir_name() << " ";
    FormatLabel(label_id);
    out_ << "(";
    FormatNodeName(arg_id);
    out_ << ")\n";
    in_terminator_sequence = false;
  }

  template <>
  auto FormatInstruction<SemanticsNode::Branch>(SemanticsNodeId,
                                                SemanticsNode node) -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    out_ << SemanticsNodeKind::Branch.ir_name() << " ";
    FormatLabel(node.GetAsBranch());
    out_ << "\n";
    in_terminator_sequence = false;
  }

  template <>
  auto FormatInstructionRHS<SemanticsNode::Call>(SemanticsNode node) -> void {
    out_ << " ";
    auto [args_id, callee_id] = node.GetAsCall();
    FormatArg(callee_id);
    FormatArg(args_id);
  }

  template <>
  auto FormatInstructionRHS<SemanticsNode::CrossReference>(SemanticsNode node)
      -> void {
    // TODO: Figure out a way to make this meaningful. We'll need some way to
    // name cross-reference IRs, perhaps by the node ID of the import?
    auto [xref_id, node_id] = node.GetAsCrossReference();
    out_ << " " << xref_id << "." << node_id;
  }

  // StructTypeFields are formatted as part of their StructType.
  template <>
  auto FormatInstruction<SemanticsNode::StructTypeField>(SemanticsNodeId,
                                                         SemanticsNode)
      -> void {}

  template <>
  auto FormatInstructionRHS<SemanticsNode::StructType>(SemanticsNode node)
      -> void {
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

  auto FormatArgs(SemanticsNode::NoArgs) -> void {}

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

  auto FormatArg(SemanticsBoolValue v) -> void { out_ << v; }

  auto FormatArg(SemanticsBuiltinKind kind) -> void { out_ << kind.label(); }

  auto FormatArg(SemanticsFunctionId id) -> void { FormatFunctionName(id); }

  auto FormatArg(SemanticsIntegerLiteralId id) -> void {
    out_ << semantics_ir_.GetIntegerLiteral(id);
  }

  auto FormatArg(SemanticsMemberIndex index) -> void { out_ << index; }

  // TODO: Should we be printing scopes inline, or should we have a separate
  // step to print them like we do for functions?
  auto FormatArg(SemanticsNameScopeId id) -> void {
    // Name scopes aren't kept in any particular order. Sort the entries before
    // we print them for stability and consistency.
    std::vector<std::pair<SemanticsNodeId, SemanticsStringId>> entries;
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

  auto FormatArg(SemanticsNodeId id) -> void { FormatNodeName(id); }

  auto FormatArg(SemanticsNodeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto node_id : semantics_ir_.GetNodeBlock(id)) {
      out_ << sep;
      FormatArg(node_id);
    }
    out_ << ')';
  }

  auto FormatArg(SemanticsRealLiteralId id) -> void {
    // TODO: Format with a `.` when the exponent is near zero.
    const auto& real = semantics_ir_.GetRealLiteral(id);
    out_ << real.mantissa << (real.is_decimal ? 'e' : 'p') << real.exponent;
  }

  auto FormatArg(SemanticsStringId id) -> void {
    out_ << '"';
    out_.write_escaped(semantics_ir_.GetString(id), /*UseHexEscapes=*/true);
    out_ << '"';
  }

  auto FormatArg(SemanticsTypeId id) -> void { FormatType(id); }

  auto FormatArg(SemanticsTypeBlockId id) -> void {
    out_ << '(';
    llvm::ListSeparator sep;
    for (auto type_id : semantics_ir_.GetTypeBlock(id)) {
      out_ << sep;
      FormatArg(type_id);
    }
    out_ << ')';
  }

  auto FormatNodeName(SemanticsNodeId id) -> void {
    out_ << node_namer_.GetNameFor(scope_, id);
  }

  auto FormatLabel(SemanticsNodeBlockId id) -> void {
    out_ << node_namer_.GetLabelFor(scope_, id);
  }

  auto FormatString(SemanticsStringId id) -> void {
    out_ << semantics_ir_.GetString(id);
  }

  auto FormatFunctionName(SemanticsFunctionId id) -> void {
    out_ << node_namer_.GetNameFor(id);
  }

  auto FormatType(SemanticsTypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << semantics_ir_.StringifyType(id, /*in_type_context=*/true);
    }
  }

 private:
  const SemanticsIR& semantics_ir_;
  llvm::raw_ostream& out_;
  NodeNamer node_namer_;
  NodeNamer::ScopeIndex scope_ = NodeNamer::ScopeIndex::None;
  bool in_terminator_sequence = false;
};

auto FormatSemanticsIR(const TokenizedBuffer& tokenized_buffer,
                       const ParseTree& parse_tree,
                       const SemanticsIR& semantics_ir, llvm::raw_ostream& out)
    -> void {
  SemanticsIRFormatter(tokenized_buffer, parse_tree, semantics_ir, out)
      .Format();
}

}  // namespace Carbon
