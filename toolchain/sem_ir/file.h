// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_FILE_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::SemIR {

// A function.
struct Function : public Printable<Function> {
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
          ", body: [{0}]",
          llvm::make_range(body_block_ids.begin(), body_block_ids.end()));
    }
    out << "}";
  }

  // The function name.
  StringId name_id;
  // A block containing a single reference node per parameter.
  NodeBlockId param_refs_id;
  // The return type. This will be invalid if the return type wasn't specified.
  TypeId return_type_id;
  // The storage for the return value, which is a reference expression whose
  // type is the return type of the function. Will be invalid if the function
  // doesn't have a return slot. If this is valid, a call to the function is
  // expected to have an additional final argument corresponding to the return
  // slot.
  NodeId return_slot_id;
  // A list of the statically reachable code blocks in the body of the
  // function, in lexical order. The first block is the entry block. This will
  // be empty for declarations that don't have a visible definition.
  llvm::SmallVector<NodeBlockId> body_block_ids;
};

struct RealLiteral : public Printable<RealLiteral> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{mantissa: ";
    mantissa.print(out, /*isSigned=*/false);
    out << ", exponent: " << exponent << ", is_decimal: " << is_decimal << "}";
  }

  llvm::APInt mantissa;
  llvm::APInt exponent;

  // If false, the value is mantissa * 2^exponent.
  // If true, the value is mantissa * 10^exponent.
  bool is_decimal;
};

// Provides semantic analysis on a Parse::Tree.
class File : public Printable<File> {
 public:
  // Produces a file for the builtins.
  explicit File();

  // Starts a new file for Check::CheckParseTree. Builtins are required.
  explicit File(std::string filename, const File* builtins);

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

  // Adds a node to the node list, returning an ID to reference the node. Note
  // that this doesn't add the node to any node block. Check::Context::AddNode
  // or NodeBlockStack::AddNode should usually be used instead, to add the node
  // to the current block.
  auto AddNodeInNoBlock(Node node) -> NodeId {
    NodeId node_id(nodes_.size());
    nodes_.push_back(node);
    return node_id;
  }

  // Overwrites a given node with a new value.
  auto ReplaceNode(NodeId node_id, Node node) -> void {
    nodes_[node_id.index] = node;
  }

  // Returns the requested node.
  auto GetNode(NodeId node_id) const -> Node { return nodes_[node_id.index]; }

  // Reserves and returns a node block ID. The contents of the node block
  // should be specified by calling SetNodeBlock, or by pushing the ID onto the
  // NodeBlockStack.
  auto AddNodeBlockId() -> NodeBlockId {
    NodeBlockId id(node_blocks_.size());
    node_blocks_.push_back({});
    return id;
  }

  // Sets the contents of an empty node block to the given content.
  auto SetNodeBlock(NodeBlockId block_id, llvm::ArrayRef<NodeId> content)
      -> void {
    CARBON_CHECK(block_id != NodeBlockId::Unreachable);
    CARBON_CHECK(node_blocks_[block_id.index].empty())
        << "node block content set more than once";
    node_blocks_[block_id.index] = AllocateCopy(content);
  }

  // Adds a node block with the given content, returning an ID to reference it.
  auto AddNodeBlock(llvm::ArrayRef<NodeId> content) -> NodeBlockId {
    NodeBlockId id(node_blocks_.size());
    node_blocks_.push_back(AllocateCopy(content));
    return id;
  }

  // Returns the requested node block.
  auto GetNodeBlock(NodeBlockId block_id) const -> llvm::ArrayRef<NodeId> {
    CARBON_CHECK(block_id != NodeBlockId::Unreachable);
    return node_blocks_[block_id.index];
  }

  // Returns the requested node block.
  auto GetNodeBlock(NodeBlockId block_id) -> llvm::MutableArrayRef<NodeId> {
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

  // Adds a type block with the given content, returning an ID to reference it.
  auto AddTypeBlock(llvm::ArrayRef<TypeId> content) -> TypeBlockId {
    TypeBlockId id(type_blocks_.size());
    type_blocks_.push_back(AllocateCopy(content));
    return id;
  }

  // Returns the requested type block.
  auto GetTypeBlock(TypeBlockId block_id) const -> llvm::ArrayRef<TypeId> {
    return type_blocks_[block_id.index];
  }

  // Returns the requested type block.
  auto GetTypeBlock(TypeBlockId block_id) -> llvm::MutableArrayRef<TypeId> {
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

  auto types() const -> llvm::ArrayRef<NodeId> { return types_; }

  auto top_node_block_id() const -> NodeBlockId { return top_node_block_id_; }
  auto set_top_node_block_id(NodeBlockId block_id) -> void {
    top_node_block_id_ = block_id;
  }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }
  auto set_has_errors(bool has_errors) -> void { has_errors_ = has_errors; }

  auto filename() const -> llvm::StringRef { return filename_; }

 private:
  // Allocates a copy of the given data using our slab allocator.
  template <typename T>
  auto AllocateCopy(llvm::ArrayRef<T> data) -> llvm::MutableArrayRef<T> {
    // We're not going to run a destructor, so ensure that's OK.
    static_assert(std::is_trivially_destructible_v<T>);

    T* storage = static_cast<T*>(
        allocator_.Allocate(data.size() * sizeof(T), alignof(T)));
    std::uninitialized_copy(data.begin(), data.end(), storage);
    return llvm::MutableArrayRef<T>(storage, data.size());
  }

  bool has_errors_ = false;

  // Slab allocator, used to allocate node and type blocks.
  llvm::BumpPtrAllocator allocator_;

  // The associated filename.
  // TODO: If SemIR starts linking back to tokens, reuse its filename.
  std::string filename_;

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

  // Type blocks within the IR. These reference entries in types_. Storage for
  // the data is provided by allocator_.
  llvm::SmallVector<llvm::MutableArrayRef<TypeId>> type_blocks_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching BuiltinKind ordering.
  llvm::SmallVector<Node> nodes_;

  // Node blocks within the IR. These reference entries in nodes_. Storage for
  // the data is provided by allocator_.
  llvm::SmallVector<llvm::MutableArrayRef<NodeId>> node_blocks_;

  // The top node block ID.
  NodeBlockId top_node_block_id_ = NodeBlockId::Invalid;
};

// The expression category of a semantics node. See /docs/design/values.md for
// details.
enum class ExpressionCategory : int8_t {
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
auto GetExpressionCategory(const File& file, NodeId node_id)
    -> ExpressionCategory;

// The value representation to use when passing by value.
struct ValueRepresentation {
  enum Kind : int8_t {
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
  TypeId type;
};

// Returns information about the value representation to use for a type.
auto GetValueRepresentation(const File& file, TypeId type_id)
    -> ValueRepresentation;

// The initializing representation to use when returning by value.
struct InitializingRepresentation {
  enum Kind : int8_t {
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
  auto has_return_slot() const -> bool { return kind == InPlace; }
};

// Returns information about the initializing representation to use for a type.
auto GetInitializingRepresentation(const File& file, TypeId type_id)
    -> InitializingRepresentation;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FILE_H_
