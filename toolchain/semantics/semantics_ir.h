// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// A call.
struct SemanticsCall {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{arg_ir: " << arg_ir_id << ", arg_refs: " << arg_refs_id << "}";
  }

  // The full IR for arguments.
  SemanticsNodeBlockId arg_ir_id;
  // A block containing a single reference node per argument.
  SemanticsNodeBlockId arg_refs_id;
};

// A callable object.
struct SemanticsCallable {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{param_ir: " << param_ir_id << ", param_refs: " << param_refs_id;
    if (return_type_id.is_valid()) {
      out << ", return_type: " << return_type_id;
    }
    out << "}";
  }

  // The full IR for parameters.
  SemanticsNodeBlockId param_ir_id;
  // A block containing a single reference node per parameter.
  SemanticsNodeBlockId param_refs_id;
  // The return type. This will be invalid if the return type wasn't specified.
  // The IR corresponding to the return type will be in a node block.
  SemanticsNodeId return_type_id;
};

struct SemanticsRealLiteral {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{mantissa: " << mantissa << ", exponent: " << exponent
        << ", is_decimal: " << is_decimal << "}";
  }

  llvm::APInt mantissa;
  llvm::APInt exponent;

  // If false, the value is mantissa * 2^exponent.
  // If true, the value is mantissa * 10^exponent.
  bool is_decimal;
};

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // Produces the builtins.
  static auto MakeBuiltinIR() -> SemanticsIR;

  // Adds the IR for the provided ParseTree.
  static auto MakeFromParseTree(const SemanticsIR& builtin_ir,
                                const TokenizedBuffer& tokens,
                                const ParseTree& parse_tree,
                                DiagnosticConsumer& consumer,
                                llvm::raw_ostream* vlog_stream) -> SemanticsIR;

  // Prints the full IR. Allow omitting builtins so that unrelated changes are
  // less likely to alternate test golden files.
  // TODO: In the future, the things to print may change, for example by adding
  // preludes. We may then want the ability to omit other things similar to
  // builtins.
  auto Print(llvm::raw_ostream& out) const -> void {
    Print(out, /*include_builtins=*/false);
  }
  auto Print(llvm::raw_ostream& out, bool include_builtins) const -> void;

  // Adds a call, returning an ID to reference it.
  auto AddCall(SemanticsCall call) -> SemanticsCallId {
    SemanticsCallId id(calls_.size());
    calls_.push_back(call);
    return id;
  }

  // Adds a callable, returning an ID to reference it.
  auto AddCallable(SemanticsCallable callable) -> SemanticsCallableId {
    SemanticsCallableId id(callables_.size());
    callables_.push_back(callable);
    return id;
  }

  // Returns the requested callable.
  auto GetCallable(SemanticsCallableId callable_id) const -> SemanticsCallable {
    return callables_[callable_id.index];
  }

  // Adds an integer literal, returning an ID to reference it.
  auto AddIntegerLiteral(llvm::APInt integer_literal)
      -> SemanticsIntegerLiteralId {
    SemanticsIntegerLiteralId id(integer_literals_.size());
    integer_literals_.push_back(integer_literal);
    return id;
  }

  // Returns the requested integer literal.
  auto GetIntegerLiteral(SemanticsIntegerLiteralId int_id) const
      -> const llvm::APInt& {
    return integer_literals_[int_id.index];
  }

  // Adds a node to a specified block, returning an ID to reference the node.
  auto AddNode(SemanticsNodeBlockId block_id, SemanticsNode node)
      -> SemanticsNodeId {
    SemanticsNodeId node_id(nodes_.size());
    nodes_.push_back(node);
    node_blocks_[block_id.index].push_back(node_id);
    return node_id;
  }

  // Returns the requested node.
  auto GetNode(SemanticsNodeId node_id) const -> SemanticsNode {
    return nodes_[node_id.index];
  }

  // Returns the requested node block.
  auto GetNodeBlock(SemanticsNodeBlockId block_id) const
      -> const llvm::SmallVector<SemanticsNodeId>& {
    return node_blocks_[block_id.index];
  }

  // Returns the requested node block.
  auto GetNodeBlock(SemanticsNodeBlockId block_id)
      -> llvm::SmallVector<SemanticsNodeId>& {
    return node_blocks_[block_id.index];
  }

  // Adds a real literal, returning an ID to reference it.
  auto AddRealLiteral(SemanticsRealLiteral real_literal)
      -> SemanticsRealLiteralId {
    SemanticsRealLiteralId id(real_literals_.size());
    real_literals_.push_back(real_literal);
    return id;
  }

  // Returns the requested real literal.
  auto GetRealLiteral(SemanticsRealLiteralId int_id) const
      -> const SemanticsRealLiteral& {
    return real_literals_[int_id.index];
  }

  // Adds an string, returning an ID to reference it.
  auto AddString(llvm::StringRef str) -> SemanticsStringId {
    // If the string has already been stored, return the corresponding ID.
    if (auto existing_id = GetStringID(str)) {
      return *existing_id;
    }

    // Allocate the string and store it in the map.
    SemanticsStringId id(strings_.size());
    strings_.push_back(str);
    CARBON_CHECK(string_to_id_.insert({str, id}).second);
    return id;
  }

  // Returns the requested string.
  auto GetString(SemanticsStringId string_id) const -> llvm::StringRef {
    return strings_[string_id.index];
  }

  // Returns an ID for the string if it's previously been stored.
  auto GetStringID(llvm::StringRef str) -> std::optional<SemanticsStringId> {
    auto str_find = string_to_id_.find(str);
    if (str_find != string_to_id_.end()) {
      return str_find->second;
    }
    return std::nullopt;
  }

  // Produces a string version of a node.
  auto StringifyNode(SemanticsNodeId node_id) -> std::string;

  auto nodes_size() const -> int { return nodes_.size(); }

  // The node blocks, for direct mutation.
  auto node_blocks() -> llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>>& {
    return node_blocks_;
  }

  auto top_node_block_id() const -> SemanticsNodeBlockId {
    return top_node_block_id_;
  }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }

 private:
  explicit SemanticsIR(const SemanticsIR* builtin_ir)
      : cross_reference_irs_({builtin_ir == nullptr ? this : builtin_ir}) {
    // For SemanticsNodeBlockId::Empty.
    node_blocks_.resize(1);
  }

  bool has_errors_ = false;

  // Storage for call objects.
  llvm::SmallVector<SemanticsCall> calls_;

  // Storage for callable objects.
  llvm::SmallVector<SemanticsCallable> callables_;

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const SemanticsIR*> cross_reference_irs_;

  // Storage for integer literals.
  llvm::SmallVector<llvm::APInt> integer_literals_;

  // Storage for real literals.
  llvm::SmallVector<SemanticsRealLiteral> real_literals_;

  // Storage for strings. strings_ provides a list of allocated strings, while
  // string_to_id_ provides a mapping to identify strings.
  llvm::StringMap<SemanticsStringId> string_to_id_;
  llvm::SmallVector<llvm::StringRef> strings_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching SemanticsBuiltinKind ordering.
  llvm::SmallVector<SemanticsNode> nodes_;

  // Storage for blocks within the IR. These reference entries in nodes_.
  llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>> node_blocks_;

  // The top node block ID.
  SemanticsNodeBlockId top_node_block_id_ = SemanticsNodeBlockId::Invalid;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
