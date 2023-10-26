// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_
#define CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_

#include "llvm/ADT/DenseMap.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::SemIR {

// Provides a ValueStore wrapper for an API specific to nodes.
class NodeStore {
 public:
  // Adds a node to the node list, returning an ID to reference the node. Note
  // that this doesn't add the node to any node block. Check::Context::AddNode
  // or NodeBlockStack::AddNode should usually be used instead, to add the node
  // to the current block.
  auto AddInNoBlock(Node node) -> NodeId { return values_.Add(node); }

  // Returns the requested node.
  auto Get(NodeId node_id) const -> Node { return values_.Get(node_id); }

  // Returns the requested node, which is known to have the specified type.
  template <typename NodeT>
  auto GetAs(NodeId node_id) const -> NodeT {
    return Get(node_id).As<NodeT>();
  }

  // Overwrites a given node with a new value.
  auto Set(NodeId node_id, Node node) -> void { values_.Get(node_id) = node; }

  // Reserves space.
  auto Reserve(size_t size) -> void { values_.Reserve(size); }

  auto array_ref() const -> llvm::ArrayRef<Node> { return values_.array_ref(); }
  auto size() const -> int { return values_.size(); }

 private:
  ValueStore<NodeId, Node> values_;
};

// Provides a ValueStore-like interface for names. This is currently a wrapper
// around a string store that has no state of its own.
class NameStore {
 public:
  NameStore(ValueStore<StringId>* strings) : strings_(strings) {}

  // Adds a string name, returning an ID to reference it.
  auto Add(StringId string_id) -> NameId { return NameId(string_id.index); }

  // Returns the requested name as a string ID, if it is a string name. This
  // returns std::nullopt for special names.
  auto GetAsStringId(NameId name_id) const -> std::optional<StringId> {
    if (name_id.index >= 0) {
      return StringId(name_id.index);
    }
    return std::nullopt;
  }

  // Returns the requested name as a string, if it is a string name. This
  // returns std::nullopt for special names.
  auto GetAsString(NameId name_id) const -> std::optional<llvm::StringRef> {
    auto id = GetAsStringId(name_id);
    if (id) {
      return strings_->Get(*id);
    }
    return std::nullopt;
  }

  // Returns the requested name as a string for formatted output. This returns
  // `"r#name"` if `name` is a keyword.
  auto GetFormatted(NameId name_id) const -> llvm::StringRef;

  // Returns a best-effort name to use as the basis for SemIR and LLVM IR
  // names. This is always an identifier, but may be ambiguous, for example if
  // there is both a `self` and an `r#self` in the same scope. Returns "" for
  // an invalid name.
  auto GetIRBaseName(NameId name_id) const -> llvm::StringRef;

 private:
  ValueStore<StringId>* strings_;
};

// Provides a ValueStore wrapper for an API specific to name scopes.
class NameScopeStore {
 public:
  // Adds a name scope, returning an ID to reference it.
  auto Add() -> NameScopeId { return values_.AddDefaultValue(); }

  // Adds an entry to a name scope. Returns true on success, false on
  // duplicates.
  auto AddEntry(NameScopeId scope_id, NameId name_id, NodeId target_id)
      -> bool {
    return values_.Get(scope_id).insert({name_id, target_id}).second;
  }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) const
      -> const llvm::DenseMap<NameId, NodeId>& {
    return values_.Get(scope_id);
  }

 private:
  ValueStore<NameScopeId, llvm::DenseMap<NameId, NodeId>> values_;
};

// Provides a block-based ValueStore, which uses slab allocation of added
// blocks. This allows references to values to outlast vector resizes that might
// otherwise invalidate references.
//
// BlockValueStore is used as-is, but there are also children that expose the
// protected members for type-specific functionality.
template <typename IdT, typename ValueT>
class BlockValueStore : public Yaml::Printable<BlockValueStore<IdT, ValueT>> {
 public:
  explicit BlockValueStore(llvm::BumpPtrAllocator& allocator)
      : allocator_(&allocator) {}

  // Adds a block with the given content, returning an ID to reference it.
  auto Add(llvm::ArrayRef<ValueT> content) -> IdT {
    return values_.Add(AllocateCopy(content));
  }

  // Returns the requested block.
  auto Get(IdT id) const -> llvm::ArrayRef<ValueT> { return values_.Get(id); }

  // Returns the requested block.
  auto Get(IdT id) -> llvm::MutableArrayRef<ValueT> { return values_.Get(id); }

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
  auto AddDefaultValue() -> NodeBlockId { return values_.AddDefaultValue(); }

  // Adds an uninitialized block of the given size.
  auto AddUninitialized(size_t size) -> NodeBlockId {
    return values_.Add(AllocateUninitialized(size));
  }

  // Sets the contents of an empty block to the given content.
  auto Set(NodeBlockId block_id, llvm::ArrayRef<NodeId> content) -> void {
    CARBON_CHECK(Get(block_id).empty())
        << "node block content set more than once";
    values_.Get(block_id) = AllocateCopy(content);
  }

 private:
  // Allocates an uninitialized array using our slab allocator.
  auto AllocateUninitialized(std::size_t size)
      -> llvm::MutableArrayRef<ValueT> {
    // We're not going to run a destructor, so ensure that's OK.
    static_assert(std::is_trivially_destructible_v<ValueT>);

    auto storage = static_cast<ValueT*>(
        allocator_->Allocate(size * sizeof(ValueT), alignof(ValueT)));
    return llvm::MutableArrayRef<ValueT>(storage, size);
  }

  // Allocates a copy of the given data using our slab allocator.
  auto AllocateCopy(llvm::ArrayRef<ValueT> data)
      -> llvm::MutableArrayRef<ValueT> {
    auto result = AllocateUninitialized(data.size());
    std::uninitialized_copy(data.begin(), data.end(), result.begin());
    return result;
  }

  llvm::BumpPtrAllocator* allocator_;
  ValueStore<IdT, llvm::MutableArrayRef<ValueT>> values_;
};

// Adapts BlockValueStore for node blocks.
class NodeBlockStore : public BlockValueStore<NodeBlockId, NodeId> {
 public:
  using BaseType = BlockValueStore<NodeBlockId, NodeId>;

  using BaseType::AddDefaultValue;
  using BaseType::AddUninitialized;
  using BaseType::BaseType;

  auto Set(NodeBlockId block_id, llvm::ArrayRef<NodeId> content) -> void {
    CARBON_CHECK(block_id != NodeBlockId::Unreachable);
    BlockValueStore<NodeBlockId, NodeId>::Set(block_id, content);
  }
};

}  // namespace Carbon::SemIR

// Support use of NameId as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::NameId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::NameId> {};

#endif  // CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_
