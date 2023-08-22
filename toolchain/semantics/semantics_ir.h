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

namespace Carbon {

// A function.
struct SemanticsFunction {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", "
        << "param_refs: " << param_refs_id;
    if (return_type_id.is_valid()) {
      out << ", return_type: " << return_type_id;
    }
    if (return_slot_id.is_valid()) {
      out << ", return_slot: " << return_slot_id;
    }
    if (!body_block_ids.empty()) {
      out << llvm::formatv(
          ", body: {{{0}}}",
          llvm::make_range(body_block_ids.begin(), body_block_ids.end()));
    }
    out << "}";
  }

  // The function name.
  SemanticsStringId name_id;
  // A block containing a single reference node per parameter.
  SemanticsNodeBlockId param_refs_id;
  // The return type. This will be invalid if the return type wasn't specified.
  SemanticsTypeId return_type_id;
  // The storage for the return value, which is a reference expression whose
  // type is the return type of the function. Will be invalid if the function
  // doesn't have a return slot. If this is valid, a call to the function is
  // expected to have an additional final argument corresponding to the return
  // slot.
  SemanticsNodeId return_slot_id;
  // A list of the statically reachable code blocks in the body of the
  // function, in lexical order. The first block is the entry block. This will
  // be empty for declarations that don't have a visible definition.
  llvm::SmallVector<SemanticsNodeBlockId> body_block_ids;
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

  // Verifies that invariants of the semantics IR hold.
  auto Verify() const -> ErrorOr<Success>;

  // Prints the full IR. Allow omitting builtins so that unrelated changes are
  // less likely to alternate test golden files.
  // TODO: In the future, the things to print may change, for example by adding
  // preludes. We may then want the ability to omit other things similar to
  // builtins.
  auto Print(llvm::raw_ostream& out) const -> void {
    Print(out, /*include_builtins=*/false);
  }
  auto Print(llvm::raw_ostream& out, bool include_builtins) const -> void;

  // Returns array bound value from the bound node.
  auto GetArrayBoundValue(SemanticsNodeId bound_id) const -> uint64_t {
    return GetIntegerLiteral(GetNode(bound_id).GetAsIntegerLiteral())
        .getZExtValue();
  }

  // Returns the requested IR.
  auto GetCrossReferenceIR(SemanticsCrossReferenceIRId xref_id) const
      -> const SemanticsIR& {
    return *cross_reference_irs_[xref_id.index];
  }

  // Adds a callable, returning an ID to reference it.
  auto AddFunction(SemanticsFunction function) -> SemanticsFunctionId {
    SemanticsFunctionId id(functions_.size());
    functions_.push_back(function);
    return id;
  }

  // Returns the requested callable.
  auto GetFunction(SemanticsFunctionId function_id) const
      -> const SemanticsFunction& {
    return functions_[function_id.index];
  }

  // Returns the requested callable.
  auto GetFunction(SemanticsFunctionId function_id) -> SemanticsFunction& {
    return functions_[function_id.index];
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

  // Adds a name scope, returning an ID to reference it.
  auto AddNameScope() -> SemanticsNameScopeId {
    SemanticsNameScopeId name_scopes_id(name_scopes_.size());
    name_scopes_.resize(name_scopes_id.index + 1);
    return name_scopes_id;
  }

  // Adds an entry to a name scope. Returns true on success, false on
  // duplicates.
  auto AddNameScopeEntry(SemanticsNameScopeId scope_id,
                         SemanticsStringId name_id, SemanticsNodeId target_id)
      -> bool {
    return name_scopes_[scope_id.index].insert({name_id, target_id}).second;
  }

  // Returns the requested name scope.
  auto GetNameScope(SemanticsNameScopeId scope_id) const
      -> const llvm::DenseMap<SemanticsStringId, SemanticsNodeId>& {
    return name_scopes_[scope_id.index];
  }

  // Adds a node to a specified block, returning an ID to reference the node.
  auto AddNode(SemanticsNodeBlockId block_id, SemanticsNode node)
      -> SemanticsNodeId {
    SemanticsNodeId node_id(nodes_.size());
    nodes_.push_back(node);
    if (block_id != SemanticsNodeBlockId::Unreachable) {
      node_blocks_[block_id.index].push_back(node_id);
    }
    return node_id;
  }

  // Overwrites a given node with a new value.
  auto ReplaceNode(SemanticsNodeId node_id, SemanticsNode node) -> void {
    nodes_[node_id.index] = node;
  }

  // Returns the requested node.
  auto GetNode(SemanticsNodeId node_id) const -> SemanticsNode {
    return nodes_[node_id.index];
  }

  // Adds an empty node block, returning an ID to reference it.
  auto AddNodeBlock() -> SemanticsNodeBlockId {
    SemanticsNodeBlockId id(node_blocks_.size());
    node_blocks_.push_back({});
    return id;
  }

  // Returns the requested node block.
  auto GetNodeBlock(SemanticsNodeBlockId block_id) const
      -> const llvm::SmallVector<SemanticsNodeId>& {
    CARBON_CHECK(block_id != SemanticsNodeBlockId::Unreachable);
    return node_blocks_[block_id.index];
  }

  // Returns the requested node block.
  auto GetNodeBlock(SemanticsNodeBlockId block_id)
      -> llvm::SmallVector<SemanticsNodeId>& {
    CARBON_CHECK(block_id != SemanticsNodeBlockId::Unreachable);
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

  // Adds a type, returning an ID to reference it.
  auto AddType(SemanticsNodeId node_id) -> SemanticsTypeId {
    SemanticsTypeId type_id(types_.size());
    types_.push_back(node_id);
    return type_id;
  }

  // Gets the node ID for a type. This doesn't handle TypeType or InvalidType in
  // order to avoid a check; callers that need that should use
  // GetTypeAllowBuiltinTypes.
  auto GetType(SemanticsTypeId type_id) const -> SemanticsNodeId {
    // Double-check it's not called with TypeType or InvalidType.
    CARBON_CHECK(type_id.index >= 0)
        << "Invalid argument for GetType: " << type_id;
    return types_[type_id.index];
  }

  auto GetTypeAllowBuiltinTypes(SemanticsTypeId type_id) const
      -> SemanticsNodeId {
    if (type_id == SemanticsTypeId::TypeType) {
      return SemanticsNodeId::BuiltinTypeType;
    } else if (type_id == SemanticsTypeId::Error) {
      return SemanticsNodeId::BuiltinError;
    } else {
      return GetType(type_id);
    }
  }

  // Adds an empty type block, returning an ID to reference it.
  auto AddTypeBlock() -> SemanticsTypeBlockId {
    SemanticsTypeBlockId id(type_blocks_.size());
    type_blocks_.push_back({});
    return id;
  }

  // Returns the requested type block.
  auto GetTypeBlock(SemanticsTypeBlockId block_id) const
      -> const llvm::SmallVector<SemanticsTypeId>& {
    return type_blocks_[block_id.index];
  }

  // Returns the requested type block.
  auto GetTypeBlock(SemanticsTypeBlockId block_id)
      -> llvm::SmallVector<SemanticsTypeId>& {
    return type_blocks_[block_id.index];
  }

  // Produces a string version of a type. If `in_type_context` is false, an
  // explicit conversion to type `type` will be added in cases where the type
  // expression would otherwise have a different type, such as a tuple or
  // struct type.
  auto StringifyType(SemanticsTypeId type_id,
                     bool in_type_context = false) const -> std::string;

  auto functions_size() const -> int { return functions_.size(); }
  auto nodes_size() const -> int { return nodes_.size(); }
  auto node_blocks_size() const -> int { return node_blocks_.size(); }

  auto types() const -> const llvm::SmallVector<SemanticsNodeId>& {
    return types_;
  }

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

  // Storage for callable objects.
  llvm::SmallVector<SemanticsFunction> functions_;

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const SemanticsIR*> cross_reference_irs_;

  // Storage for integer literals.
  llvm::SmallVector<llvm::APInt> integer_literals_;

  // Storage for name scopes.
  llvm::SmallVector<llvm::DenseMap<SemanticsStringId, SemanticsNodeId>>
      name_scopes_;

  // Storage for real literals.
  llvm::SmallVector<SemanticsRealLiteral> real_literals_;

  // Storage for strings. strings_ provides a list of allocated strings, while
  // string_to_id_ provides a mapping to identify strings.
  llvm::StringMap<SemanticsStringId> string_to_id_;
  llvm::SmallVector<llvm::StringRef> strings_;

  // Nodes which correspond to in-use types. Stored separately for easy access
  // by lowering.
  llvm::SmallVector<SemanticsNodeId> types_;

  // Storage for blocks within the IR. These reference entries in types_.
  llvm::SmallVector<llvm::SmallVector<SemanticsTypeId>> type_blocks_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching SemanticsBuiltinKind ordering.
  llvm::SmallVector<SemanticsNode> nodes_;

  // Storage for blocks within the IR. These reference entries in nodes_.
  llvm::SmallVector<llvm::SmallVector<SemanticsNodeId>> node_blocks_;

  // The top node block ID.
  SemanticsNodeBlockId top_node_block_id_ = SemanticsNodeBlockId::Invalid;
};

// The expression category of a semantics node. See /docs/design/values.md for
// details.
enum class SemanticsExpressionCategory {
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
auto GetSemanticsExpressionCategory(const SemanticsIR& semantics_ir,
                                    SemanticsNodeId node_id)
    -> SemanticsExpressionCategory;

// The value representation to use when passing by value.
struct SemanticsValueRepresentation {
  enum Kind {
    // The type has no value representation. This is used for empty types, such
    // as `()`, where there is no value.
    None,
    // The value representation is a copy of the value. On call boundaries, the
    // value itself will be passed. `type` is the value type.
    // TODO: `type` should be `const`-qualified, but is currently not.
    Copy,
    // The value representation is a pointer to an object. When used as a
    // parameter, the argument is a reference expression. `type` is the pointee
    // type.
    // TODO: `type` should be `const`-qualified, but is currently not.
    Pointer,
    // The value representation has been customized, and has the same behavior
    // as the value representation of some other type.
    // TODO: This is not implemented or used yet.
    Custom,
  };
  // The kind of value representation used by this type.
  Kind kind;
  // The type used to model the value representation.
  SemanticsTypeId type;
};

// Returns information about the value representation to use for a type.
auto GetSemanticsValueRepresentation(const SemanticsIR& semantics_ir,
                                     SemanticsTypeId type_id)
    -> SemanticsValueRepresentation;

// The initializing representation to use when returning by value.
struct SemanticsInitializingRepresentation {
  enum Kind {
    // The type has no initializing representation. This is used for empty
    // types, where no initialization is necessary.
    None,
    // An initializing expression produces a value, which is copied into the
    // initialized object.
    ByCopy,
    // An initializing expression takes a location as input, which is
    // initialized as a side effect of evaluating the expression.
    InPlace,
    // TODO: Consider adding a kind where the expression takes an advisory
    // location and returns a value plus an indicator of whether the location
    // was actually initialized.
  };
  // The kind of initializing representation used by this type.
  Kind kind;

  // Returns whether a return slot is used when returning this type.
  bool has_return_slot() const { return kind == InPlace; }
};

// Returns information about the initializing representation to use for a type.
auto GetSemanticsInitializingRepresentation(const SemanticsIR& semantics_ir,
                                            SemanticsTypeId type_id)
    -> SemanticsInitializingRepresentation;

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
