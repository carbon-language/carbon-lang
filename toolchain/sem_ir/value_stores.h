// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_
#define CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_

#include <type_traits>

#include "llvm/ADT/DenseMap.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type_info.h"

namespace Carbon::SemIR {

// Associates a NodeId and Inst in order to provide type-checking that the
// TypedNodeId corresponds to the InstT.
struct ParseNodeAndInst {
  // In cases where the NodeId is untyped or the InstT is unknown, the check
  // can't be done at compile time.
  // TODO: Consider runtime validation that InstT::Kind::TypedNodeId
  // corresponds.
  static auto Untyped(Parse::NodeId parse_node, Inst inst) -> ParseNodeAndInst {
    return ParseNodeAndInst(parse_node, inst, /*is_untyped=*/true);
  }

  // For the common case, support construction as:
  //   context.AddInst({parse_node, SemIR::MyInst{...}});
  template <typename InstT>
    requires(Internal::HasParseNode<InstT>)
  // NOLINTNEXTLINE(google-explicit-constructor)
  ParseNodeAndInst(decltype(InstT::Kind)::TypedNodeId parse_node, InstT inst)
      : parse_node(parse_node), inst(inst) {}

  // For cases with no parse node, support construction as:
  //   context.AddInst({SemIR::MyInst{...}});
  template <typename InstT>
    requires(!Internal::HasParseNode<InstT>)
  // NOLINTNEXTLINE(google-explicit-constructor)
  ParseNodeAndInst(InstT inst)
      : parse_node(Parse::NodeId::Invalid), inst(inst) {}

  Parse::NodeId parse_node;
  Inst inst;

 private:
  explicit ParseNodeAndInst(Parse::NodeId parse_node, Inst inst,
                            bool /*is_untyped*/)
      : parse_node(parse_node), inst(inst) {}
};

// Provides a ValueStore wrapper for an API specific to instructions.
class InstStore {
 public:
  // Adds an instruction to the instruction list, returning an ID to reference
  // the instruction. Note that this doesn't add the instruction to any
  // instruction block. Check::Context::AddInst or InstBlockStack::AddInst
  // should usually be used instead, to add the instruction to the current
  // block.
  auto AddInNoBlock(ParseNodeAndInst parse_node_and_inst) -> InstId {
    parse_nodes_.push_back(parse_node_and_inst.parse_node);
    return values_.Add(parse_node_and_inst.inst);
  }

  // Returns the requested instruction.
  auto Get(InstId inst_id) const -> Inst { return values_.Get(inst_id); }

  // Returns the requested instruction and its parse node.
  auto GetWithParseNode(InstId inst_id) const -> ParseNodeAndInst {
    return ParseNodeAndInst::Untyped(GetParseNode(inst_id), Get(inst_id));
  }

  // Returns the requested instruction, which is known to have the specified
  // type.
  template <typename InstT>
  auto GetAs(InstId inst_id) const -> InstT {
    return Get(inst_id).As<InstT>();
  }

  // Returns the requested instruction as the specified type, if it is of that
  // type.
  template <typename InstT>
  auto TryGetAs(InstId inst_id) const -> std::optional<InstT> {
    return Get(inst_id).TryAs<InstT>();
  }

  auto GetParseNode(InstId inst_id) const -> Parse::NodeId {
    return parse_nodes_[inst_id.index];
  }

  // Overwrites a given instruction and parse node with a new value.
  auto Set(InstId inst_id, ParseNodeAndInst parse_node_and_inst) -> void {
    values_.Get(inst_id) = parse_node_and_inst.inst;
    parse_nodes_[inst_id.index] = parse_node_and_inst.parse_node;
  }

  auto SetParseNode(InstId inst_id, Parse::NodeId parse_node) -> void {
    parse_nodes_[inst_id.index] = parse_node;
  }

  // Reserves space.
  auto Reserve(size_t size) -> void {
    parse_nodes_.reserve(size);
    values_.Reserve(size);
  }

  auto array_ref() const -> llvm::ArrayRef<Inst> { return values_.array_ref(); }
  auto size() const -> int { return values_.size(); }

 private:
  llvm::SmallVector<Parse::NodeId> parse_nodes_;
  ValueStore<InstId> values_;
};

// Provides a ValueStore wrapper for tracking the constant values of
// instructions.
class ConstantValueStore {
 public:
  // Returns the constant value of the requested instruction, or
  // `ConstantId::NotConstant` if it is not constant.
  auto Get(InstId inst_id) const -> ConstantId {
    CARBON_CHECK(inst_id.index >= 0);
    return static_cast<size_t>(inst_id.index) >= values_.size()
               ? ConstantId::NotConstant
               : values_[inst_id.index];
  }

  // Sets the constant value of the given instruction, or sets that it is known
  // to not be a constant.
  auto Set(InstId inst_id, ConstantId const_id) -> void {
    CARBON_CHECK(inst_id.index >= 0);
    if (static_cast<size_t>(inst_id.index) >= values_.size()) {
      values_.resize(inst_id.index + 1, ConstantId::NotConstant);
    }
    values_[inst_id.index] = const_id;
  }

 private:
  // A mapping from `InstId::index` to the corresponding constant value. This is
  // expected to be sparse, and may be smaller than the list of instructions if
  // there are trailing non-constant instructions.
  //
  // Set inline size to 0 because these will typically be too large for the
  // stack, while this does make File smaller.
  llvm::SmallVector<ConstantId, 0> values_;
};

// Provides storage for instructions representing deduplicated global constants.
class ConstantStore {
 public:
  explicit ConstantStore(File& sem_ir, llvm::BumpPtrAllocator& allocator)
      : allocator_(&allocator), constants_(&sem_ir) {}

  // Adds a new constant instruction, or gets the existing constant with this
  // value. Returns the ID of the constant.
  //
  // This updates `sem_ir.insts()` and `sem_ir.constant_values()` if the
  // constant is new.
  auto GetOrAdd(Inst inst, bool is_symbolic) -> ConstantId;

  // Returns a copy of the constant IDs as a vector, in an arbitrary but
  // stable order. This should not be used anywhere performance-sensitive.
  auto GetAsVector() const -> llvm::SmallVector<InstId, 0>;

  auto size() const -> int { return constants_.size(); }

 private:
  // TODO: We store two copies of each constant instruction: one in insts() and
  // one here. We could avoid one of those copies and store just an InstId here,
  // at the cost of some more indirection when recomputing profiles during
  // lookup. Once we have a representative data set, we should measure the
  // impact on compile time from that change.
  struct ConstantNode : llvm::FoldingSetNode {
    Inst inst;
    ConstantId constant_id;

    auto Profile(llvm::FoldingSetNodeID& id, File* sem_ir) -> void;
  };

  llvm::BumpPtrAllocator* allocator_;
  llvm::ContextualFoldingSet<ConstantNode, File*> constants_;
};

// Provides a ValueStore wrapper with an API specific to types.
class TypeStore : public ValueStore<TypeId> {
 public:
  explicit TypeStore(InstStore* insts) : insts_(insts) {}

  // Returns the ID of the constant used to define the specified type.
  auto GetConstantId(TypeId type_id) const -> ConstantId {
    if (type_id == TypeId::TypeType) {
      return ConstantId::ForTemplateConstant(InstId::BuiltinTypeType);
    } else if (type_id == TypeId::Error) {
      return ConstantId::Error;
    } else if (!type_id.is_valid()) {
      // TODO: Can we CHECK-fail on this?
      return ConstantId::NotConstant;
    } else {
      return Get(type_id).constant_id;
    }
  }

  // Returns the ID of the instruction used to define the specified type.
  auto GetInstId(TypeId type_id) const -> InstId {
    return GetConstantId(type_id).inst_id();
  }

  // Returns the instruction used to define the specified type.
  auto GetAsInst(TypeId type_id) const -> Inst {
    return insts_->Get(GetInstId(type_id));
  }

  // Returns the instruction used to define the specified type, which is known
  // to be a particular kind of instruction.
  template <typename InstT>
  auto GetAs(TypeId type_id) const -> InstT {
    if constexpr (std::is_same_v<InstT, Builtin>) {
      return GetAsInst(type_id).As<InstT>();
    } else {
      // The type is not a builtin, so no need to check for special values.
      return insts_->Get(Get(type_id).constant_id.inst_id()).As<InstT>();
    }
  }

  // Returns the instruction used to define the specified type, if it is of a
  // particular kind.
  template <typename InstT>
  auto TryGetAs(TypeId type_id) const -> std::optional<InstT> {
    return GetAsInst(type_id).TryAs<InstT>();
  }

  // Gets the value representation to use for a type. This returns an
  // invalid type if the given type is not complete.
  auto GetValueRepr(TypeId type_id) const -> ValueRepr {
    if (type_id.index < 0) {
      // TypeType and InvalidType are their own value representation.
      return {.kind = ValueRepr::Copy, .type_id = type_id};
    }
    return Get(type_id).value_repr;
  }

  // Determines whether the given type is known to be complete. This does not
  // determine whether the type could be completed, only whether it has been.
  auto IsComplete(TypeId type_id) const -> bool {
    return GetValueRepr(type_id).kind != ValueRepr::Unknown;
  }

 private:
  InstStore* insts_;
};

// Provides a ValueStore-like interface for names.
//
// A name is either an identifier name or a special name such as `self` that
// does not correspond to an identifier token. Identifier names are represented
// as `NameId`s with the same non-negative index as the `IdentifierId` of the
// identifier. Special names are represented as `NameId`s with a negative index.
//
// `SemIR::NameId` values should be obtained by using `NameId::ForIdentifier`
// or the named constants such as `NameId::SelfValue`.
//
// As we do not require any additional explicit storage for names, this is
// currently a wrapper around an identifier store that has no state of its own.
class NameStoreWrapper {
 public:
  explicit NameStoreWrapper(const StringStoreWrapper<IdentifierId>* identifiers)
      : identifiers_(identifiers) {}

  // Returns the requested name as a string, if it is an identifier name. This
  // returns std::nullopt for special names.
  auto GetAsStringIfIdentifier(NameId name_id) const
      -> std::optional<llvm::StringRef> {
    if (auto identifier_id = name_id.AsIdentifierId();
        identifier_id.is_valid()) {
      return identifiers_->Get(identifier_id);
    }
    return std::nullopt;
  }

  // Returns the requested name as a string for formatted output. This returns
  // `"r#name"` if `name` is a keyword.
  auto GetFormatted(NameId name_id) const -> llvm::StringRef;

  // Returns a best-effort name to use as the basis for SemIR and LLVM IR names.
  // This is always identifier-shaped, but may be ambiguous, for example if
  // there is both a `self` and an `r#self` in the same scope. Returns "" for an
  // invalid name.
  auto GetIRBaseName(NameId name_id) const -> llvm::StringRef;

 private:
  const StringStoreWrapper<IdentifierId>* identifiers_;
};

struct NameScope : Printable<NameScope> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // Names in the scope.
  llvm::DenseMap<NameId, InstId> names = llvm::DenseMap<NameId, InstId>();

  // Scopes extended by this scope.
  //
  // TODO: A `NameScopeId` is currently insufficient to describe an extended
  // scope in general. For example:
  //
  //   class A(T:! type) {
  //     extend base: B(T*);
  //   }
  //
  // needs to describe the `T*` argument.
  //
  // Small vector size is set to 1: we expect that there will rarely be more
  // than a single extended scope. Currently the only kind of extended scope is
  // a base class, and there can be only one of those per scope.
  // TODO: Revisit this once we have more kinds of extended scope and data.
  // TODO: Consider using something like `TinyPtrVector` for this.
  llvm::SmallVector<NameScopeId, 1> extended_scopes;

  // The instruction which owns the scope.
  InstId inst_id;

  // When the scope is a namespace, the name. Otherwise, invalid.
  NameId name_id;

  // The scope enclosing this one.
  NameScopeId enclosing_scope_id;

  // Whether we have diagnosed an error in a construct that would have added
  // names to this scope. For example, this can happen if an `import` failed or
  // an `extend` declaration was ill-formed. If true, the `names` map is assumed
  // to be missing names as a result of the error, and no further errors are
  // produced for lookup failures in this scope.
  bool has_error = false;
};

// Provides a ValueStore wrapper for an API specific to name scopes.
class NameScopeStore {
 public:
  // Adds a name scope, returning an ID to reference it.
  auto Add(InstId inst_id, NameId name_id, NameScopeId enclosing_scope_id)
      -> NameScopeId {
    return values_.Add({.inst_id = inst_id,
                        .name_id = name_id,
                        .enclosing_scope_id = enclosing_scope_id});
  }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) -> NameScope& { return values_.Get(scope_id); }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) const -> const NameScope& {
    return values_.Get(scope_id);
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

 private:
  ValueStore<NameScopeId> values_;
};

// Provides a block-based ValueStore, which uses slab allocation of added
// blocks. This allows references to values to outlast vector resizes that might
// otherwise invalidate references.
//
// BlockValueStore is used as-is, but there are also children that expose the
// protected members for type-specific functionality.
//
// On IdT, this requires:
//   - IdT::ElementType to represent the underlying type in the block.
//   - IdT::ValueType to be llvm::MutableArrayRef<IdT::ElementType> for
//     compatibility with ValueStore.
template <typename IdT>
class BlockValueStore : public Yaml::Printable<BlockValueStore<IdT>> {
 public:
  using ElementType = IdT::ElementType;

  explicit BlockValueStore(llvm::BumpPtrAllocator& allocator)
      : allocator_(&allocator) {}

  // Adds a block with the given content, returning an ID to reference it.
  auto Add(llvm::ArrayRef<ElementType> content) -> IdT {
    return values_.Add(AllocateCopy(content));
  }

  // Returns the requested block.
  auto Get(IdT id) const -> llvm::ArrayRef<ElementType> {
    return values_.Get(id);
  }

  // Returns the requested block.
  auto Get(IdT id) -> llvm::MutableArrayRef<ElementType> {
    return values_.Get(id);
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
      for (auto block_index : llvm::seq(values_.size())) {
        auto block_id = IdT(block_index);
        map.Add(PrintToString(block_id),
                Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                  auto block = Get(block_id);
                  for (auto i : llvm::seq(block.size())) {
                    map.Add(llvm::itostr(i), Yaml::OutputScalar(block[i]));
                  }
                }));
      }
    });
  }

  auto size() const -> int { return values_.size(); }

 protected:
  // Reserves and returns a block ID. The contents of the block
  // should be specified by calling Set, or similar.
  auto AddDefaultValue() -> InstBlockId { return values_.AddDefaultValue(); }

  // Adds an uninitialized block of the given size.
  auto AddUninitialized(size_t size) -> InstBlockId {
    return values_.Add(AllocateUninitialized(size));
  }

  // Sets the contents of an empty block to the given content.
  auto Set(InstBlockId block_id, llvm::ArrayRef<InstId> content) -> void {
    CARBON_CHECK(Get(block_id).empty())
        << "inst block content set more than once";
    values_.Get(block_id) = AllocateCopy(content);
  }

 private:
  // Allocates an uninitialized array using our slab allocator.
  auto AllocateUninitialized(std::size_t size)
      -> llvm::MutableArrayRef<ElementType> {
    // We're not going to run a destructor, so ensure that's OK.
    static_assert(std::is_trivially_destructible_v<ElementType>);

    auto storage = static_cast<ElementType*>(
        allocator_->Allocate(size * sizeof(ElementType), alignof(ElementType)));
    return llvm::MutableArrayRef<ElementType>(storage, size);
  }

  // Allocates a copy of the given data using our slab allocator.
  auto AllocateCopy(llvm::ArrayRef<ElementType> data)
      -> llvm::MutableArrayRef<ElementType> {
    auto result = AllocateUninitialized(data.size());
    std::uninitialized_copy(data.begin(), data.end(), result.begin());
    return result;
  }

  llvm::BumpPtrAllocator* allocator_;
  ValueStore<IdT> values_;
};

// Adapts BlockValueStore for instruction blocks.
class InstBlockStore : public BlockValueStore<InstBlockId> {
 public:
  using BaseType = BlockValueStore<InstBlockId>;

  using BaseType::AddDefaultValue;
  using BaseType::AddUninitialized;

  explicit InstBlockStore(llvm::BumpPtrAllocator& allocator)
      : BaseType(allocator) {
    auto empty_id = AddDefaultValue();
    CARBON_CHECK(empty_id == InstBlockId::Empty);
    auto exports_id = AddDefaultValue();
    CARBON_CHECK(exports_id == InstBlockId::Exports);
  }

  auto Set(InstBlockId block_id, llvm::ArrayRef<InstId> content) -> void {
    CARBON_CHECK(block_id != InstBlockId::Unreachable);
    BlockValueStore<InstBlockId>::Set(block_id, content);
  }
};

}  // namespace Carbon::SemIR

// Support use of NameId as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::NameId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::NameId> {};

#endif  // CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_
