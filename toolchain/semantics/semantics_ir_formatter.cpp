// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_formatter.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

namespace {
class NodeNamer {
 public:
  auto GetNameFor(SemanticsNodeId node_id) -> std::string {
    auto it = names.find(node_id);
    if (it == names.end()) {
      // This should not happen in valid IR.
      return "<noderef " + llvm::itostr(node_id.index) + ">";
    }
    return it->second;
  }

  auto GetLabelFor(SemanticsNodeBlockId block_id) -> std::string {
    auto it = labels.find(block_id);
    if (it == labels.end()) {
      // This should not happen in valid IR.
      return "<nodeblockref " + llvm::itostr(block_id.index) + ">";
    }
    return it->second;
  }

  auto AddBlockLabel(SemanticsNodeBlockId block_id) -> void {
    if (label_count == 0) {
      labels[block_id] = "!entry";
    } else {
      labels[block_id] = "!" + llvm::itostr(label_count - 1);
    }
    ++label_count;
  }

  auto CollectNamesInBlock(const SemanticsIR& semantics_ir,
                           SemanticsNodeBlockId block_id) -> void {
    // Use bound names where available.
    for (auto node_id : semantics_ir.GetNodeBlock(block_id)) {
      auto node = semantics_ir.GetNode(node_id);
      if (node.kind() == SemanticsNodeKind::BindName) {
        auto [name_id, named_node_id] = node.GetAsBindName();
        names[named_node_id] = ("%" + semantics_ir.GetString(name_id)).str();
      }
    }

    // Sequentially number all remaining typed values.
    for (auto node_id : semantics_ir.GetNodeBlock(block_id)) {
      auto node = semantics_ir.GetNode(node_id);
      if (node.kind().type_field_kind() == SemanticsTypeFieldKind::Type) {
        auto& name = names[node_id];
        if (name.empty()) {
          names[node_id] = "%" + llvm::itostr(unnamed_count++);
        }
      }
    }
  }

 private:
  llvm::DenseMap<SemanticsNodeId, std::string> names;
  llvm::DenseMap<SemanticsNodeBlockId, std::string> labels;
  int unnamed_count = 0;
  int label_count = 0;
};
}

class SemanticsIRFormatter {
 public:
  class NodeNameScope {
   public:
    NodeNameScope(SemanticsIRFormatter& parent) : parent_(parent) {
      CARBON_CHECK(!parent_.node_name_scope) << "multiple name scopes at once";
      parent_.node_name_scope = &namer_;
    }
    ~NodeNameScope() {
      parent_.node_name_scope = nullptr;
    }

    auto AddBlockLabel(SemanticsNodeBlockId block_id) -> void {
      namer_.AddBlockLabel(block_id);
    }

    auto CollectNamesInBlock(SemanticsNodeBlockId block_id) -> void {
      namer_.CollectNamesInBlock(parent_.semantics_ir_, block_id);
    }

   private:
    SemanticsIRFormatter& parent_;
    NodeNamer namer_;
  };

  explicit SemanticsIRFormatter(const SemanticsIR& semantics_ir,
                                llvm::raw_ostream& out)
      : semantics_ir_(semantics_ir), out_(out) {}

  auto Format() -> void {
    // TODO: Include more information from the package declaration.
    out_ << "package {\n";
    // TODO: Handle the case where there are multiple top-level node blocks.
    // For example, there may be branching in the initializer of a global or a
    // type expression.
    {
      NodeNameScope package_scope(*this);
      package_scope.CollectNamesInBlock(semantics_ir_.top_node_block_id());
      FormatCodeBlock(semantics_ir_.top_node_block_id());
    }
    out_ << "}\n";

    for (int i = 0; i != semantics_ir_.functions_size(); ++i) {
      FormatFunction(SemanticsFunctionId(i));
    }
  }

  auto FormatFunction(SemanticsFunctionId id) -> void {
    FormatFunction(semantics_ir_.GetFunction(id));
  }

  auto FormatFunction(const SemanticsFunction &fn) -> void {
    out_ << "\nfn ";
    FormatGlobalName(fn.name_id);
    out_ << "(";

    // Assign names to values and blocks in this function.
    NodeNameScope function_scope(*this);
    function_scope.CollectNamesInBlock(fn.param_refs_id);
    for (auto block_id : fn.body_block_ids) {
      function_scope.AddBlockLabel(block_id);
      function_scope.CollectNamesInBlock(block_id);
    }

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
      out_ << ";";
    }
  }

  auto FormatCodeBlock(SemanticsNodeBlockId block_id) -> void {
    for (const SemanticsNodeId node_id : semantics_ir_.GetNodeBlock(block_id)) {
      FormatInstruction(node_id);
    }
  }

  auto FormatInstruction(SemanticsNodeId node_id) -> void {
    if (!node_id.is_valid()) {
      out_ << "  InvalidInstruction\n";
      return;
    }

    FormatInstruction(node_id, semantics_ir_.GetNode(node_id));
  }

  auto FormatInstruction(SemanticsNodeId node_id, SemanticsNode node) -> void {
    switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)                         \
  case SemanticsNodeKind::Name:                                  \
    FormatInstruction<SemanticsNode::Name>(node_id, node, #Name, \
                                           node.GetAs##Name());  \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
  }

  template <typename Kind>
  auto FormatInstruction(SemanticsNodeId node_id, SemanticsNode node,
                         llvm::StringRef name, decltype(Kind::Get(node)) args)
      -> void {
    out_ << "  ";
    FormatInstructionLHS(node_id, node);
    out_ << name;
    if (node.kind().type_field_kind() == SemanticsTypeFieldKind::Argument) {
      FormatArgs(std::pair(node.type_id(), args));
    } else {
      FormatArgs(args);
    }
    out_ << "\n";
  }

  auto FormatInstructionLHS(SemanticsNodeId node_id, SemanticsNode node)
      -> void {
    if (node.kind().type_field_kind() == SemanticsTypeFieldKind::Type) {
      FormatNodeName(node_id);
      out_ << ": ";
      FormatType(node.type_id());
      out_ << " = ";
    }
  }

  // BindName is handled by the NodeNamer and doesn't appear in the output.
  template <>
  auto FormatInstruction<SemanticsNode::BindName>(
      SemanticsNodeId, SemanticsNode, llvm::StringRef,
      std::pair<SemanticsStringId, SemanticsNodeId>) -> void {}

  template <>
  auto FormatInstruction<SemanticsNode::BlockArg>(SemanticsNodeId node_id,
                                                      SemanticsNode node,
                                                      llvm::StringRef,
                                                      SemanticsNodeBlockId self)
      -> void {
    out_ << "  ";
    FormatInstructionLHS(node_id, node);
    out_ << "block_arg ";
    FormatLabel(self);
    out_ << "\n";
  }

  template <>
  auto FormatInstruction<SemanticsNode::BranchIf>(
      SemanticsNodeId, SemanticsNode, llvm::StringRef,
      std::pair<SemanticsNodeBlockId, SemanticsNodeId> args) -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    out_ << "if ";
    FormatNodeName(args.second);
    out_ << " br ";
    FormatLabel(args.first);
    out_ << " else ";
    in_terminator_sequence = true;
  }

  template <>
  auto FormatInstruction<SemanticsNode::BranchWithArg>(
      SemanticsNodeId, SemanticsNode, llvm::StringRef,
      std::pair<SemanticsNodeBlockId, SemanticsNodeId> args) -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    out_ << "br ";
    FormatLabel(args.first);
    out_ << "(";
    FormatNodeName(args.second);
    out_ << ")\n";
    in_terminator_sequence = false;
  }

  template <>
  auto FormatInstruction<SemanticsNode::Branch>(
      SemanticsNodeId, SemanticsNode, llvm::StringRef,
      SemanticsNodeBlockId target) -> void {
    if (!in_terminator_sequence) {
      out_ << "  ";
    }
    out_ << "br ";
    FormatLabel(target);
    out_ << "\n";
    in_terminator_sequence = false;
  }

  auto FormatArgs(SemanticsNode::NoArgs) -> void {}

  template<typename Arg1>
  auto FormatArgs(Arg1 arg) -> void {
    out_ << ' ';
    FormatArg(arg);
  }

  template<typename Arg1, typename Arg2>
  auto FormatArgs(std::pair<Arg1, Arg2> args) -> void {
    out_ << ' ';
    FormatArg(args.first);
    out_ << ",";
    FormatArgs(args.second);
  }

  // TODO: Replace this with custom formatting.
  template<typename T>
  auto FormatArg(T v) -> void {
    out_ << v;
  }

  auto FormatArg(SemanticsFunctionId id) -> void {
    FormatGlobalName(semantics_ir_.GetFunction(id).name_id);
  }

  auto FormatArg(SemanticsIntegerLiteralId id) -> void {
    out_ << semantics_ir_.GetIntegerLiteral(id);
  }

  auto FormatArg(SemanticsNodeId id) -> void {
    out_ << node_name_scope->GetNameFor(id);
  }

  auto FormatArg(SemanticsNodeBlockId id) -> void {
    out_ << '{';
    llvm::ListSeparator sep;
    for (auto node_id : semantics_ir_.GetNodeBlock(id)) {
      out_ << sep;
      FormatArg(node_id);
    }
    out_ << '}';
  }

  auto FormatArg(SemanticsRealLiteralId id) -> void {
    // TODO: Format with a `.` when the exponent is near zero.
    const auto & real = semantics_ir_.GetRealLiteral(id);
    out_ << real.mantissa << (real.is_decimal ? 'e' : 'p') << real.exponent;
  }

  auto FormatArg(SemanticsStringId id) -> void {
    out_ << '"';
    out_.write_escaped(semantics_ir_.GetString(id), /*UseHexEscapes=*/true);
    out_ << '"';
  }

  auto FormatArg(SemanticsTypeId id) -> void {
    FormatType(id);
  }

  auto FormatArg(SemanticsTypeBlockId id) -> void {
    out_ << '{';
    llvm::ListSeparator sep;
    for (auto type_id : semantics_ir_.GetTypeBlock(id)) {
      out_ << sep;
      FormatArg(type_id);
    }
    out_ << '}';
  }

  auto FormatNodeName(SemanticsNodeId id) -> void {
    out_ << node_name_scope->GetNameFor(id);
  }

  auto FormatLabel(SemanticsNodeBlockId id) -> void {
    out_ << node_name_scope->GetLabelFor(id);
  }

  auto FormatString(SemanticsStringId id) -> void {
    out_ << semantics_ir_.GetString(id);
  }

  auto FormatGlobalName(SemanticsStringId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << '@';
      FormatString(id);
    }
  }

  auto FormatType(SemanticsTypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << semantics_ir_.StringifyType(id);
    }
  }

 private:
  const SemanticsIR& semantics_ir_;
  llvm::raw_ostream& out_;
  NodeNamer* node_name_scope = nullptr;
  bool in_terminator_sequence = false;
};

auto FormatSemanticsIR(const SemanticsIR& ir, llvm::raw_ostream& out) -> void {
  SemanticsIRFormatter(ir, out).Format();
}

}  // namespace Carbon
