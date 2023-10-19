// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_FILE_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/value_store.h"
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
  // The definition, if the function has been defined or is currently being
  // defined. This is a FunctionDeclaration.
  NodeId definition_id = NodeId::Invalid;
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
  llvm::SmallVector<NodeBlockId> body_block_ids = {};
};

// A class.
struct Class : public Printable<Class> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id;
    out << "}";
  }

  // The class name.
  StringId name_id;

  // The definition, if the class has been defined or is currently being
  // defined. This is a ClassDeclaration.
  NodeId definition_id = NodeId::Invalid;

  // The class scope.
  NameScopeId scope_id = NameScopeId::Invalid;

  // The first block of the class body.
  // TODO: Handle control flow in the class body, such as if-expressions.
  NodeBlockId body_block_id = NodeBlockId::Invalid;
};

// The value representation to use when passing by value.
struct ValueRepresentation : public Printable<ValueRepresentation> {
  auto Print(llvm::raw_ostream& out) const -> void;

  enum Kind : int8_t {
    // The value representation is not yet known. This is used for incomplete
    // types.
    Unknown,
    // The type has no value representation. This is used for empty types, such
    // as `()`, where there is no value.
    None,
    // The value representation is a copy of the value. On call boundaries, the
    // value itself will be passed. `type` is the value type.
    Copy,
    // The value representation is a pointer to an object. When used as a
    // parameter, the argument is a reference expression. `type` is the pointee
    // type.
    Pointer,
    // The value representation has been customized, and has the same behavior
    // as the value representation of some other type.
    // TODO: This is not implemented or used yet.
    Custom,
  };
  // The kind of value representation used by this type.
  Kind kind = Unknown;
  // The type used to model the value representation.
  TypeId type_id = TypeId::Invalid;
};

// Information stored about a TypeId.
struct TypeInfo : public Printable<TypeInfo> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // The node that defines this type.
  NodeId node_id;
  // The value representation for this type. Will be `Unknown` if the type is
  // not complete.
  ValueRepresentation value_representation = ValueRepresentation();
};

// Provides semantic analysis on a Parse::Tree.
class File : public Printable<File> {
 public:
  // Produces a file for the builtins.
  explicit File(SharedValueStores& value_stores);

  // Starts a new file for Check::CheckParseTree. Builtins are required.
  explicit File(SharedValueStores& value_stores, std::string filename,
                const File* builtins);

  // Verifies that invariants of the semantics IR hold.
  auto Verify() const -> ErrorOr<Success>;

  // Prints the full IR. Allow omitting builtins so that unrelated changes are
  // less likely to alter test golden files.
  // TODO: In the future, the things to print may change, for example by adding
  // preludes. We may then want the ability to omit other things similar to
  // builtins.
  auto Print(llvm::raw_ostream& out, bool include_builtins) const -> void;

  auto Print(llvm::raw_ostream& out) const -> void {
    Print(out, /*include_builtins=*/false);
  }

  // Returns array bound value from the bound node.
  auto GetArrayBoundValue(NodeId bound_id) const -> uint64_t {
    return integers()
        .Get(GetNodeAs<IntegerLiteral>(bound_id).integer_id)
        .getZExtValue();
  }

  // Returns the requested IR.
  auto GetCrossReferenceIR(CrossReferenceIRId xref_id) const -> const File& {
    return *cross_reference_irs_[xref_id.index];
  }

  // Adds a name scope, returning an ID to reference it.
  auto AddNameScope() -> NameScopeId {
    NameScopeId name_scopes_id(name_scopes_.size());
    // TODO: Return failure on overflow instead of crashing.
    CARBON_CHECK(name_scopes_id.index >= 0);
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
    // TODO: Return failure on overflow instead of crashing.
    CARBON_CHECK(node_id.index >= 0);
    nodes_.push_back(node);
    return node_id;
  }

  // Overwrites a given node with a new value.
  auto ReplaceNode(NodeId node_id, Node node) -> void {
    nodes_[node_id.index] = node;
  }

  // Returns the requested node.
  auto GetNode(NodeId node_id) const -> Node { return nodes_[node_id.index]; }

  // Returns the requested node, which is known to have the specified type.
  template <typename NodeT>
  auto GetNodeAs(NodeId node_id) const -> NodeT {
    return GetNode(node_id).As<NodeT>();
  }

  // Reserves and returns a node block ID. The contents of the node block
  // should be specified by calling SetNodeBlock, or by pushing the ID onto the
  // NodeBlockStack.
  auto AddNodeBlockId() -> NodeBlockId {
    NodeBlockId id(node_blocks_.size());
    // TODO: Return failure on overflow instead of crashing.
    CARBON_CHECK(id.index >= 0);
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
    // TODO: Return failure on overflow instead of crashing.
    CARBON_CHECK(id.index >= 0);
    node_blocks_.push_back(AllocateCopy(content));
    return id;
  }

  // Adds a node block of the given size.
  auto AddUninitializedNodeBlock(size_t size) -> NodeBlockId {
    NodeBlockId id(node_blocks_.size());
    // TODO: Return failure on overflow instead of crashing.
    CARBON_CHECK(id.index >= 0);
    node_blocks_.push_back(AllocateUninitialized<NodeId>(size));
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

  // Marks a type as complete, and sets its value representation.
  auto CompleteType(TypeId object_type_id,
                    ValueRepresentation value_representation) -> void {
    if (object_type_id.index < 0) {
      // We already know our builtin types are complete.
      return;
    }
    CARBON_CHECK(types().Get(object_type_id).value_representation.kind ==
                 ValueRepresentation::Unknown)
        << "Type " << object_type_id << " completed more than once";
    types().Get(object_type_id).value_representation = value_representation;
  }

  auto GetTypeAllowBuiltinTypes(TypeId type_id) const -> NodeId {
    if (type_id == TypeId::TypeType) {
      return NodeId::BuiltinTypeType;
    } else if (type_id == TypeId::Error) {
      return NodeId::BuiltinError;
    } else if (type_id == TypeId::Invalid) {
      return NodeId::Invalid;
    } else {
      return types().Get(type_id).node_id;
    }
  }

  // Gets the value representation to use for a type. This returns an
  // invalid type if the given type is not complete.
  auto GetValueRepresentation(TypeId type_id) const -> ValueRepresentation {
    if (type_id.index < 0) {
      // TypeType and InvalidType are their own value representation.
      return {.kind = ValueRepresentation::Copy, .type_id = type_id};
    }
    return types().Get(type_id).value_representation;
  }

  // Determines whether the given type is known to be complete. This does not
  // determine whether the type could be completed, only whether it has been.
  auto IsTypeComplete(TypeId type_id) const -> bool {
    return GetValueRepresentation(type_id).kind != ValueRepresentation::Unknown;
  }

  // Gets the pointee type of the given type, which must be a pointer type.
  auto GetPointeeType(TypeId pointer_id) const -> TypeId {
    return GetNodeAs<PointerType>(types().Get(pointer_id).node_id).pointee_id;
  }

  // Adds a type block with the given content, returning an ID to reference it.
  auto AddTypeBlock(llvm::ArrayRef<TypeId> content) -> TypeBlockId {
    TypeBlockId id(type_blocks_.size());
    // TODO: Return failure on overflow instead of crashing.
    CARBON_CHECK(id.index >= 0);
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

  // Same as `StringifyType`, but starting with a node representing a type
  // expression rather than a canonical type.
  auto StringifyTypeExpression(NodeId outer_node_id,
                               bool in_type_context = false) const
      -> std::string;

  // Directly expose SharedValueStores members.
  auto integers() -> ValueStore<IntegerId>& {
    return value_stores_->integers();
  }
  auto integers() const -> const ValueStore<IntegerId>& {
    return value_stores_->integers();
  }
  auto reals() const -> const ValueStore<RealId>& {
    return value_stores_->reals();
  }
  auto strings() -> ValueStore<StringId>& { return value_stores_->strings(); }
  auto strings() const -> const ValueStore<StringId>& {
    return value_stores_->strings();
  }

  auto functions() -> ValueStore<FunctionId, Function>& { return functions_; }
  auto functions() const -> const ValueStore<FunctionId, Function>& {
    return functions_;
  }
  auto classes() -> ValueStore<ClassId, Class>& { return classes_; }
  auto classes() const -> const ValueStore<ClassId, Class>& { return classes_; }
  auto types() -> ValueStore<TypeId, TypeInfo>& { return types_; }
  auto types() const -> const ValueStore<TypeId, TypeInfo>& { return types_; }

  auto nodes_size() const -> int { return nodes_.size(); }
  auto node_blocks_size() const -> int { return node_blocks_.size(); }

  auto top_node_block_id() const -> NodeBlockId { return top_node_block_id_; }
  auto set_top_node_block_id(NodeBlockId block_id) -> void {
    top_node_block_id_ = block_id;
  }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }
  auto set_has_errors(bool has_errors) -> void { has_errors_ = has_errors; }

  auto filename() const -> llvm::StringRef { return filename_; }

 private:
  // Allocates an uninitialized array using our slab allocator.
  template <typename T>
  auto AllocateUninitialized(std::size_t size) -> llvm::MutableArrayRef<T> {
    // We're not going to run a destructor, so ensure that's OK.
    static_assert(std::is_trivially_destructible_v<T>);

    T* storage =
        static_cast<T*>(allocator_.Allocate(size * sizeof(T), alignof(T)));
    return llvm::MutableArrayRef<T>(storage, size);
  }

  // Allocates a copy of the given data using our slab allocator.
  template <typename T>
  auto AllocateCopy(llvm::ArrayRef<T> data) -> llvm::MutableArrayRef<T> {
    auto result = AllocateUninitialized<T>(data.size());
    std::uninitialized_copy(data.begin(), data.end(), result.begin());
    return result;
  }

  bool has_errors_ = false;

  // Shared, compile-scoped values.
  SharedValueStores* value_stores_;

  // Slab allocator, used to allocate node and type blocks.
  llvm::BumpPtrAllocator allocator_;

  // The associated filename.
  // TODO: If SemIR starts linking back to tokens, reuse its filename.
  std::string filename_;

  // Storage for callable objects.
  ValueStore<FunctionId, Function> functions_;

  // Storage for classes.
  ValueStore<ClassId, Class> classes_;

  // Related IRs. There will always be at least 2 entries, the builtin IR (used
  // for references of builtins) followed by the current IR (used for references
  // crossing node blocks).
  llvm::SmallVector<const File*> cross_reference_irs_;

  // Storage for name scopes.
  llvm::SmallVector<llvm::DenseMap<StringId, NodeId>> name_scopes_;

  // Descriptions of types used in this file.
  ValueStore<TypeId, TypeInfo> types_;

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
  // The category of this node is not known due to an error.
  Error,
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
  // This node represents a syntactic combination of expressions that are
  // permitted to have different expression categories. This is used for tuple
  // and struct literals, where the subexpressions for different elements can
  // have different categories.
  Mixed,
  Last = Mixed
};

// Returns the expression category for a node.
auto GetExpressionCategory(const File& file, NodeId node_id)
    -> ExpressionCategory;

// Returns information about the value representation to use for a type.
inline auto GetValueRepresentation(const File& file, TypeId type_id)
    -> ValueRepresentation {
  return file.GetValueRepresentation(type_id);
}

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
