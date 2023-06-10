// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_formatter.h"

namespace Carbon {

class SemanticsIRFormatter {
 public:
  explicit SemanticsIRFormatter(const SemanticsIR& semantics_ir,
                                llvm::raw_ostream& out)
      : semantics_ir_(semantics_ir), out_(out) {}

  auto Format() -> void {
    // TODO: Include more information from the package declaration.
    out_ << "package {\n";
    // TODO: Handle the case where there are multiple top-level node blocks.
    // For example, there may be branching in the initializer of a global or a
    // type expression.
    FormatCodeBlock(semantics_ir_.top_node_block_id());
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

    llvm::ListSeparator sep;
    for (const SemanticsNodeId param_id :
         semantics_ir_.GetNodeBlock(fn.param_refs_id)) {
      out_ << sep;
      auto param = semantics_ir_.GetNode(param_id);
      auto [name_id, node_id] = param.GetAsBindName();
      FormatBindingName(name_id);
      out_ << ": ";
      FormatType(param.type_id());
    }
    out_ << ")";
    if (fn.return_type_id.is_valid()) {
      out_ << " -> ";
      FormatType(fn.return_type_id);
    }

    if (!fn.body_block_ids.empty()) {
      out_ << " {\n";
      for (auto block_id : fn.body_block_ids) {
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
      out_ << "\n";
    }
  }

  auto FormatInstruction(SemanticsNodeId node_id) -> void {
    out_ << "  ";
    if (!node_id.is_valid()) {
      out_ << "InvalidInstruction";
      return;
    }

    auto node = semantics_ir_.GetNode(node_id);

    if (node.kind().type_field_kind() == SemanticsTypeFieldKind::Type) {
      out_ << "%" << node_id.index;
      out_ << ": ";
      FormatType(node.type_id());
      out_ << " = ";
    }
    FormatInstruction(node);
  }

  auto FormatInstruction(SemanticsNode insn) -> void {
    switch (insn.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)                                     \
  case SemanticsNodeKind::Name:                                              \
    out_ << #Name;                                                           \
    if (insn.kind().type_field_kind() == SemanticsTypeFieldKind::Argument) { \
      FormatArgs(std::pair(insn.type_id(), insn.GetAs##Name()));             \
    } else {                                                                 \
      FormatArgs(insn.GetAs##Name());                                        \
    }                                                                        \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
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
    out_ << '%' << id.index;
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

  auto FormatBindingName(SemanticsStringId id) -> void {
    out_ << '%';
    FormatString(id);
  }

  auto FormatType(SemanticsTypeId id) -> void {
    if (!id.is_valid()) {
      out_ << "invalid";
    } else {
      out_ << semantics_ir_.StringifyType(id);
    }
  }

 private:
  const SemanticsIR &semantics_ir_;
  llvm::raw_ostream &out_;
};

auto FormatSemanticsIR(const SemanticsIR& ir, llvm::raw_ostream& out) -> void {
  SemanticsIRFormatter(ir, out).Format();
}

}  // namespace Carbon
