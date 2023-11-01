// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_FILE_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/value_stores.h"

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
  // A block containing a single reference node per implicit parameter.
  NodeBlockId implicit_param_refs_id;
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

  // Determines whether this class has been fully defined. This is false until
  // we reach the `}` of the class definition.
  bool is_defined() const { return object_representation_id.is_valid(); }

  // The following members always have values, and do not change throughout the
  // lifetime of the class.

  // The class name.
  StringId name_id;
  // The class type, which is the type of `Self` in the class definition.
  TypeId self_type_id;
  // The first declaration of the class. This is a ClassDeclaration.
  NodeId declaration_id = NodeId::Invalid;

  // The following members are set at the `{` of the class definition.

  // The definition of the class. This is a ClassDeclaration.
  NodeId definition_id = NodeId::Invalid;
  // The class scope.
  NameScopeId scope_id = NameScopeId::Invalid;
  // The first block of the class body.
  // TODO: Handle control flow in the class body, such as if-expressions.
  NodeBlockId body_block_id = NodeBlockId::Invalid;

  // The following members are set at the `}` of the class definition.

  // The object representation type to use for this class. This is valid once
  // the class is defined.
  TypeId object_representation_id = TypeId::Invalid;
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
    // The value representation is a pointer to the value. When used as a
    // parameter, the argument is a reference expression. `type` is the pointee
    // type.
    Pointer,
    // The value representation has been customized, and has the same behavior
    // as the value representation of some other type.
    // TODO: This is not implemented or used yet.
    Custom,
  };

  enum AggregateKind : int8_t {
    // This type is not an aggregation of other types.
    NotAggregate,
    // This type is an aggregate that holds the value representations of its
    // elements.
    ValueAggregate,
    // This type is an aggregate that holds the object representations of its
    // elements.
    ObjectAggregate,
    // This type is an aggregate for which the value and object representation
    // of all elements are the same, so it effectively holds both.
    ValueAndObjectAggregate,
  };

  // Returns whether this is an aggregate that holds its elements by value.
  auto elements_are_values() const {
    return aggregate_kind == ValueAggregate ||
           aggregate_kind == ValueAndObjectAggregate;
  }

  // The kind of value representation used by this type.
  Kind kind = Unknown;
  // The kind of aggregate representation used by this type.
  AggregateKind aggregate_kind = AggregateKind::NotAggregate;
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
  auto Print(llvm::raw_ostream& out, bool include_builtins = false) const
      -> void {
    Yaml::Print(out, OutputYaml(include_builtins));
  }
  auto OutputYaml(bool include_builtins) const -> Yaml::OutputMapping;

  // Returns array bound value from the bound node.
  auto GetArrayBoundValue(NodeId bound_id) const -> uint64_t {
    return integers()
        .Get(nodes().GetAs<IntegerLiteral>(bound_id).integer_id)
        .getZExtValue();
  }

  // Returns the requested IR.
  auto GetCrossReferenceIR(CrossReferenceIRId xref_id) const -> const File& {
    return *cross_reference_irs_[xref_id.index];
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
    complete_types_.push_back(object_type_id);
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
    return nodes()
        .GetAs<PointerType>(types().Get(pointer_id).node_id)
        .pointee_id;
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
  auto reals() -> ValueStore<RealId>& { return value_stores_->reals(); }
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
  auto name_scopes() -> NameScopeStore& { return name_scopes_; }
  auto name_scopes() const -> const NameScopeStore& { return name_scopes_; }
  auto types() -> ValueStore<TypeId, TypeInfo>& { return types_; }
  auto types() const -> const ValueStore<TypeId, TypeInfo>& { return types_; }
  auto type_blocks() -> BlockValueStore<TypeBlockId, TypeId>& {
    return type_blocks_;
  }
  auto type_blocks() const -> const BlockValueStore<TypeBlockId, TypeId>& {
    return type_blocks_;
  }
  auto nodes() -> NodeStore& { return nodes_; }
  auto nodes() const -> const NodeStore& { return nodes_; }
  auto node_blocks() -> NodeBlockStore& { return node_blocks_; }
  auto node_blocks() const -> const NodeBlockStore& { return node_blocks_; }

  // A list of types that were completed in this file, in the order in which
  // they were completed. Earlier types in this list cannot contain instances of
  // later types.
  auto complete_types() const -> llvm::ArrayRef<TypeId> {
    return complete_types_;
  }

  auto top_node_block_id() const -> NodeBlockId { return top_node_block_id_; }
  auto set_top_node_block_id(NodeBlockId block_id) -> void {
    top_node_block_id_ = block_id;
  }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }
  auto set_has_errors(bool has_errors) -> void { has_errors_ = has_errors; }

  auto filename() const -> llvm::StringRef { return filename_; }

 private:
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
  NameScopeStore name_scopes_;

  // Descriptions of types used in this file.
  ValueStore<TypeId, TypeInfo> types_;

  // Types that were completed in this file.
  llvm::SmallVector<TypeId> complete_types_;

  // Type blocks within the IR. These reference entries in types_. Storage for
  // the data is provided by allocator_.
  BlockValueStore<TypeBlockId, TypeId> type_blocks_;

  // All nodes. The first entries will always be cross-references to builtins,
  // at indices matching BuiltinKind ordering.
  NodeStore nodes_;

  // Node blocks within the IR. These reference entries in nodes_. Storage for
  // the data is provided by allocator_.
  NodeBlockStore node_blocks_;

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
    // An initializing expression produces an object representation by value,
    // which is copied into the initialized object.
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
