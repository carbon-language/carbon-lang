// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon::SemIR {

// A function.
struct Function {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", "
        << "param_refs: " << param_refs_id;
    if (return_type_id.is_valid()) {
      out << ", return_type: " << return_type_id;
    }
    if (!body_block_ids.empty()) {
      out << llvm::formatv(
          ", body: [{0}]",
          llvm::make_range(body_block_ids.begin(), body_block_ids.end()));
    }
    out << "}";
  }
  LLVM_DUMP_METHOD void Dump() const { llvm::errs() << *this; }

  // The function name.
  StringId name_id;
  // A block containing a single reference node per parameter.
  NodeBlockId param_refs_id;
  // The return type. This will be invalid if the return type wasn't specified.
  TypeId return_type_id;
  // A list of the statically reachable code blocks in the body of the
  // function, in lexical order. The first block is the entry block. This will
  // be empty for declarations that don't have a visible definition.
  llvm::SmallVector<NodeBlockId> body_block_ids;
};

struct RealLiteral {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{mantissa: " << mantissa << ", exponent: " << exponent
        << ", is_decimal: " << is_decimal << "}";
  }
  LLVM_DUMP_METHOD void Dump() const { llvm::errs() << *this; }

  llvm::APInt mantissa;
  llvm::APInt exponent;

  // If false, the value is mantissa * 2^exponent.
  // If true, the value is mantissa * 10^exponent.
  bool is_decimal;
};

// Provides semantic analysis on a ParseTree.
class File {
 public:
  // Produces the builtins.
  static auto MakeBuiltinIR() -> File;

  // Adds the IR for the provided ParseTree.
  static auto MakeFromParseTree(const File& builtin_ir,
                                const TokenizedBuffer& tokens,
                                const ParseTree& parse_tree,
                                DiagnosticConsumer& consumer,
                                llvm::raw_ostream* vlog_stream) -> File;

  // Verifies that invariants of the semantics IR hold.
  auto Verify() const -> ErrorOr<Success>;

  // Prints the full IR. Allow omitting builtins so that unrelated changes are
  // less likely to alternate test golden files.
  // TODO: In the future, the things to print may change, for example by adding
  // preludes. We may then want the ability to omit other things similar to
  // builtins.
  auto Print(llvm::raw_ostream& out, bool include_builtins) const -> void;

  auto Print(llvm::raw_ostream& out) const -> void {
    Print(out, /*include_builtins=*/false);
  }

  // Returns array bound value from the bound node.
  auto GetArrayBoundValue(NodeId bound_id) const -> uint64_t {
    return GetIntegerLiteral(GetNode(bound_id).GetAsIntegerLiteral())
        .getZExtValue();
  }

  // Returns the requested IR.
  auto GetCrossReferenceIR(CrossReferenceIRId xref_id) const -> const File& {
    return *cross_reference_irs_[xref_id.index];
  }

  // Adds a callable, returning an ID to reference it.
  auto AddFunction(Function function) -> FunctionId {
    FunctionId id(functions_.size());
    functions_.push_back(function);
    return id;
  }

  // Returns the requested callable.
  auto GetFunction(FunctionId function_id) const -> const Function& {
    return functions_[function_id.index];
  }

  // Returns the requested callable.
  auto GetFunction(FunctionId function_id) -> Function& {
    return functions_[function_id.index];
  }

  // Adds an integer literal, returning an ID to reference it.
  auto AddIntegerLiteral(llvm::APInt integer_literal) -> IntegerLiteralId {
    IntegerLiteralId id(integer_literals_.size());
    integer_literals_.push_back(integer_literal);
    return id;
  }

  // Returns the requested integer literal.
  auto GetIntegerLiteral(IntegerLiteralId int_id) const -> const llvm::APInt& {
    return integer_literals_[int_id.index];
  }

  // Adds a name scope, returning an ID to reference it.
  auto AddNameScope() -> NameScopeId {
    NameScopeId name_scopes_id(name_scopes_.size());
    name_scopes_.resize(name_scopes_id.index + 1);
    return name_scopes_id;
  }

  // Adds an entry to a name scope. Returns true on success, false on
  // duplicates.
  auto AddNameScopeEntry(NameScopeId scope_id, StringId name_id,
                         NodeId target_id) -> bool {
    return name_scopes_[scope_id.index].insert({name_id, target_id}).second;
  }

  // Returns the requested name scope.
  auto GetNameScope(NameScopeId scope_id) const
      -> const llvm::DenseMap<StringId, NodeId>& {
    return name_scopes_[scope_id.index];
  }

  // Adds a node to a specified block, returning an ID to reference the node.
  auto AddNode(NodeBlockId block_id, Node node) -> NodeId {
    NodeId node_id(nodes_.size());
    nodes_.push_back(node);
    if (block_id != NodeBlockId::Unreachable) {
      node_blocks_[block_id.index].push_back(node_id);
    }
    return node_id;
  }

  // Returns the requested node.
  auto GetNode(NodeId node_id) const -> Node { return nodes_[node_id.index]; }

  // Adds an empty node block, returning an ID to reference it.
  auto AddNodeBlock() -> NodeBlockId {
    NodeBlockId id(node_blocks_.size());
    node_blocks_.push_back({});
    return id;
  }

  // Returns the requested node block.
  auto GetNodeBlock(NodeBlockId block_id) const
      -> const llvm::SmallVector<NodeId>& {
    CARBON_CHECK(block_id != NodeBlockId::Unreachable);
    return node_blocks_[block_id.index];
  }

  // Returns the requested node block.
  auto GetNodeBlock(NodeBlockId block_id) -> llvm::SmallVector<NodeId>& {
    CARBON_CHECK(block_id != NodeBlockId::Unreachable);
    return node_blocks_[block_id.index];
  }

  // Adds a real literal, returning an ID to reference it.
  auto AddRealLiteral(RealLiteral real_literal) -> RealLiteralId {
    RealLiteralId id(real_literals_.size());
    real_literals_.push_back(real_literal);
    return id;
  }

  // Returns the requested real literal.
  auto GetRealLiteral(RealLiteralId int_id) const -> const RealLiteral& {
    return real_literals_[int_id.index];
  }

  // Adds an string, returning an ID to reference it.
  auto AddString(llvm::StringRef str) -> StringId {
    // Look up the string, or add it if it's new.
    StringId next_id(strings_.size());
    auto [it, added] = string_to_id_.insert({str, next_id});

    if (added) {
      // Update the reverse mapping from IDs to strings.
      CARBON_CHECK(it->second == next_id);
      strings_.push_back(it->first());
    }

    return it->second;
  }

  // Returns the requested string.
  auto GetString(StringId string_id) const -> llvm::StringRef {
    return strings_[string_id.index];
  }

  // Adds a type, returning an ID to reference it.
  auto AddType(NodeId node_id) -> TypeId {
    TypeId type_id(types_.size());
    types_.push_back(node_id);
    return type_id;
  }

  // Gets the node ID for a type. This doesn't handle TypeType or InvalidType in
  // order to avoid a check; callers that need that should use
  // GetTypeAllowBuiltinTypes.
  auto GetType(TypeId type_id) const -> NodeId {
    // Double-check it's not called with TypeType or InvalidType.
    CARBON_CHECK(type_id.index >= 0)
        << "Invalid argument for GetType: " << type_id;
    return types_[type_id.index];
  }

  auto GetTypeAllowBuiltinTypes(TypeId type_id) const -> NodeId {
    if (type_id == TypeId::TypeType) {
      return NodeId::BuiltinTypeType;
    } else if (type_id == TypeId::Error) {
      return NodeId::BuiltinError;
    } else {
      return GetType(type_id);
    }
  }

  // Adds an empty type block, returning an ID to reference it.
  auto AddTypeBlock() -> TypeBlockId {
    TypeBlockId id(type_blocks_.size());
    type_blocks_.push_back({});
    return id;
  }

  // Returns the requested type block.
  auto GetTypeBlock(TypeBlockId block_id) const
      -> const llvm::SmallVector<TypeId>& {
    return type_blocks_[block_id.index];
  }

  // Returns the requested type block.
  auto GetTypeBlock(TypeBlockId block_id) -> llvm::SmallVector<TypeId>& {
    return type_blocks_[block_id.index];
  }

  // Produces a string version of a type. If `in_type_context` is false, an
  // explicit conversion to type `type` will be added in cases where the type
  // expression would otherwise have a different type, such as a tuple or
  // struct type.
  auto StringifyType(TypeId type_id, bool in_type_context = false) const
      -> std::string;

  auto functions_size() const -> int { return functions_.size(); }
  auto nodes_size() const -> int { return nodes_.size(); }
  auto node_blocks_size() const -> int { return node_blocks_.size(); }

  auto types() const -> const llvm::SmallVector<NodeId>& { return types_; }

  // The node blocks, for direct mutation.
  auto node_blocks() -> llvm::SmallVector<llvm::SmallVector<NodeId>>& {
    return node_blocks_;
  }

  auto top_node_block_id() const -> NodeBlockId { return top_node_block_id_; }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }

 private:
  explicit File(const File* builtin_ir)
      : cross_reference_irs_({builtin_ir == nullptr ? this : builtin_ir}) {
    // For NodeBlockId::Empty.
    node_blocks_.resize(1);
  }

  bool has_errors_ = false;

  // Storage for callable objects.
  llvm::SmallVector<Function> functions_;

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const File*> cross_reference_irs_;

  // Storage for integer literals.
  llvm::SmallVector<llvm::APInt> integer_literals_;

  // Storage for name scopes.
  llvm::SmallVector<llvm::DenseMap<StringId, NodeId>> name_scopes_;

  // Storage for real literals.
  llvm::SmallVector<RealLiteral> real_literals_;

  // Storage for strings. strings_ provides a list of allocated strings, while
  // string_to_id_ provides a mapping to identify strings.
  llvm::StringMap<StringId> string_to_id_;
  llvm::SmallVector<llvm::StringRef> strings_;

  // Nodes which correspond to in-use types. Stored separately for easy access
  // by lowering.
  llvm::SmallVector<NodeId> types_;

  // Storage for blocks within the IR. These reference entries in types_.
  llvm::SmallVector<llvm::SmallVector<TypeId>> type_blocks_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching BuiltinKind ordering.
  llvm::SmallVector<Node> nodes_;

  // Storage for blocks within the IR. These reference entries in nodes_.
  llvm::SmallVector<llvm::SmallVector<NodeId>> node_blocks_;

  // The top node block ID.
  NodeBlockId top_node_block_id_ = NodeBlockId::Invalid;
};

// The expression category of a semantics node. See /docs/design/values.md for
// details.
enum class ExpressionCategory {
  // This node does not correspond to an expression, and as such has no
  // category.
  NotExpression,
  // This node represents a value expression.
  Value,
  // This node represents a durable reference expression, that denotes an
  // object that outlives the current full expression context.
  DurableReference,
  // This node represents an ephemeral reference expression, that denotes an
  // object that does not outlive the current full expression context.
  EphemeralReference,
  // This node represents an initializing expression, that describes how to
  // initialize an object.
  Initializing,
};

// Returns the expression category for a node.
auto GetExpressionCategory(const File& semantics_ir, NodeId node_id)
    -> ExpressionCategory;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
