// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_
#define CARBON_TOOLCHAIN_SEM_IR_VALUE_STORES_H_

#include "llvm/ADT/DenseMap.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type_info.h"

namespace Carbon::SemIR {

// Provides a ValueStore wrapper for an API specific to instructions.
class InstStore {
 public:
  // Adds an instruction to the instruction list, returning an ID to reference
  // the instruction. Note that this doesn't add the instruction to any
  // instruction block. Check::Context::AddInst or InstBlockStack::AddInst
  // should usually be used instead, to add the instruction to the current
  // block.
  auto AddInNoBlock(Inst inst) -> InstId { return values_.Add(inst); }

  // Returns the requested instruction.
  auto Get(InstId inst_id) const -> Inst { return values_.Get(inst_id); }

  // Returns the requested instruction, which is known to have the specified
  // type.
  template <typename InstT>
  auto GetAs(InstId inst_id) const -> InstT {
    return Get(inst_id).As<InstT>();
  }

  // Overwrites a given instruction with a new value.
  auto Set(InstId inst_id, Inst inst) -> void { values_.Get(inst_id) = inst; }

  // Reserves space.
  auto Reserve(size_t size) -> void { values_.Reserve(size); }

  auto array_ref() const -> llvm::ArrayRef<Inst> { return values_.array_ref(); }
  auto size() const -> int { return values_.size(); }

 private:
  ValueStore<InstId> values_;
};

// Provides storage for instructions representing global constants.
class ConstantStore {
 public:
  // Add a constant instruction.
  auto Add(InstId inst_id) -> void { values_.push_back(inst_id); }

  auto array_ref() const -> llvm::ArrayRef<InstId> { return values_; }
  auto size() const -> int { return values_.size(); }

 private:
  llvm::SmallVector<InstId> values_;
};

// Provides a ValueStore wrapper with an API specific to types.
class TypeStore : public ValueStore<TypeId> {
 public:
  explicit TypeStore(InstStore* insts) : insts_(insts) {}

  // Returns the ID of the instruction used to define the specified type.
  auto GetInstId(TypeId type_id) const -> InstId {
    if (type_id == TypeId::TypeType) {
      return InstId::BuiltinTypeType;
    } else if (type_id == TypeId::Error) {
      return InstId::BuiltinError;
    } else if (type_id == TypeId::Invalid) {
      return InstId::Invalid;
    } else {
      return Get(type_id).inst_id;
    }
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
      return insts_->Get(Get(type_id).inst_id).As<InstT>();
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
// identifier. Special names are represented as `NameId`s with a negative
// index.
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

  // Returns a best-effort name to use as the basis for SemIR and LLVM IR
  // names. This is always identifier-shaped, but may be ambiguous, for example
  // if there is both a `self` and an `r#self` in the same scope. Returns ""
  // for an invalid name.
  auto GetIRBaseName(NameId name_id) const -> llvm::StringRef;

 private:
  const StringStoreWrapper<IdentifierId>* identifiers_;
};

// Provides a ValueStore wrapper for an API specific to name scopes.
class NameScopeStore {
 public:
  // Adds a name scope, returning an ID to reference it.
  auto Add() -> NameScopeId { return values_.AddDefaultValue(); }

  // Adds an entry to a name scope. Returns true on success, false on
  // duplicates.
  auto AddEntry(NameScopeId scope_id, NameId name_id, InstId target_id)
      -> bool {
    return values_.Get(scope_id).insert({name_id, target_id}).second;
  }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) const
      -> const llvm::DenseMap<NameId, InstId>& {
    return values_.Get(scope_id);
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
  using ElementType = typename IdT::ElementType;

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
  using BaseType::BaseType;

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
