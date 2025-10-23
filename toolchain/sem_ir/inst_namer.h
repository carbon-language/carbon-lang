// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_NAMER_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_NAMER_H_

#include "common/type_enum.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/inst_fingerprinter.h"

namespace Carbon::SemIR {

// Assigns names to instructions, blocks, and scopes in the Semantics IR.
//
// If `<unexpected>` occurs in output of valid SemIR, it often means the
// instruction needs to be handled by `NamingContext::NameInst` (see the cpp
// file). Note that `<unexpected>` can occur in invalid SemIR just because we're
// unable to correctly walk the SemIR.
class InstNamer {
 public:
  // int32_t matches the input value size.
  enum class ScopeId : int32_t {
    None = -1,
    // The three top-level scopes.
    File = 0,
    Imports = 1,
    Constants = 2,
    // The first entity scope; see entities in `ScopeIdTypeEnum`.
    FirstEntityScope = 3,
  };
  static_assert(sizeof(ScopeId) == sizeof(AnyIdBase));

  // Entities whose scopes get entries from `ScopeId`.
  using ScopeIdTypeEnum =
      TypeEnum<AssociatedConstantId, ClassId, CppOverloadSetId, FunctionId,
               ImplId, InterfaceId, NamedConstraintId, SpecificInterfaceId,
               VtableId>;

  // Construct the instruction namer, and assign names to all instructions in
  // the provided file.
  explicit InstNamer(const File* sem_ir, int total_ir_count);

  // Returns the scope ID corresponding to an ID of a function, class, or
  // interface.
  template <typename IdT>
    requires ScopeIdTypeEnum::Contains<IdT>
  auto GetScopeFor(IdT id) const -> ScopeId {
    auto index = id.index;
    if constexpr (std::is_same_v<IdT, ClassId>) {
      index = sem_ir_->classes().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, CppOverloadSetId>) {
      index = sem_ir_->cpp_overload_sets().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, AssociatedConstantId>) {
      index = sem_ir_->associated_constants().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, FunctionId>) {
      index = sem_ir_->functions().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, ImplId>) {
      index = sem_ir_->impls().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, InterfaceId>) {
      index = sem_ir_->interfaces().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, SpecificInterfaceId>) {
      index = sem_ir_->specific_interfaces().GetRawIndex(id);
    } else if constexpr (std::is_same_v<IdT, VtableId>) {
      index = sem_ir_->vtables().GetRawIndex(id);
    }
    return static_cast<ScopeId>(GetScopeIdOffset(ScopeIdTypeEnum::For<IdT>) +
                                index);
  }

  // Returns the scope ID corresponding to a generic. A generic object shares
  // its scope with its generic entity.
  auto GetScopeFor(GenericId id) const -> ScopeId {
    return generic_scopes_[sem_ir_->generics().GetRawIndex(id)];
  }

  // Returns the IR name for the specified scope.
  auto GetScopeName(ScopeId scope) const -> std::string;

  // Returns the name for a parent NameScope. Does not return a name for
  // namespaces. Used as part of naming functions with their containing scope.
  auto GetNameForParentNameScope(NameScopeId name_scope_id) -> llvm::StringRef;

  // Returns the IR name to use for a function, class, or interface.
  template <typename IdT>
    requires(ScopeIdTypeEnum::Contains<IdT> || std::same_as<IdT, GenericId>)
  auto GetNameFor(IdT id) const -> std::string {
    if (!id.has_value()) {
      return "invalid";
    }
    return GetScopeName(GetScopeFor(id));
  }

  // Returns the IR name to use for an instruction within its own scope, without
  // any prefix. Returns an empty string if there isn't a good name.
  auto GetUnscopedNameFor(InstId inst_id) const -> llvm::StringRef;

  // Returns the IR name to use for an instruction, when referenced from a given
  // scope.
  auto GetNameFor(ScopeId scope_id, InstId inst_id) const -> std::string;

  // Returns the IR name to use for a label within its own scope, without any
  // prefix. Returns an empty string if there isn't a good name.
  auto GetUnscopedLabelFor(InstBlockId block_id) const -> llvm::StringRef;

  // Returns the IR name to use for a label, when referenced from a given scope.
  auto GetLabelFor(ScopeId scope_id, InstBlockId block_id) const -> std::string;

  // Returns true if the instruction has a specific name assigned.
  auto has_name(InstId inst_id) const -> bool;

 private:
  // A space in which unique names can be allocated.
  class Namespace {
   private:
    // A result of a name lookup.
    struct NameResult;

   public:
    // A name in a namespace, which might be redirected to refer to another name
    // for disambiguation purposes.
    class Name {
     public:
      explicit Name() : value_(nullptr) {}
      explicit Name(llvm::StringMap<NameResult>::iterator it,
                    size_t base_name_size)
          : value_(&*it), base_name_size_(base_name_size) {}

      explicit operator bool() const { return value_; }

      // Returns the disambiguated name.
      auto GetFullName() const -> llvm::StringRef;

      // Returns the base name, without any disambiguators.
      auto GetBaseName() const -> llvm::StringRef;

      auto SetFallback(Name name) -> void { value_->second.fallback = name; }

      auto SetAmbiguous() -> void { value_->second.ambiguous = true; }

     private:
      llvm::StringMapEntry<NameResult>* value_;

      // The base name length within `value_->first()`.
      size_t base_name_size_;
    };

    // Allocates and returns a name, handling ambiguity.
    auto AllocateName(const InstNamer& inst_namer,
                      std::variant<LocId, uint64_t> loc_id_or_fingerprint,
                      std::string name) -> Name;

   private:
    struct NameResult {
      bool ambiguous = false;
      Name fallback = Name();
    };

    llvm::StringMap<NameResult> allocated_;
  };

  // A named scope that contains named entities.
  struct Scope {
    Namespace::Name name;
    Namespace insts;
    Namespace labels;
  };

  // Helper class for naming a single instruction.
  class NamingContext;

  auto GetScopeInfo(ScopeId scope_id) -> Scope& {
    return scopes_[static_cast<int>(scope_id)];
  }

  auto GetScopeInfo(ScopeId scope_id) const -> const Scope& {
    return scopes_[static_cast<int>(scope_id)];
  }

  // For the given `IdT`, returns its start offset in the `ScopeId` space. Each
  // of `ScopeIdTypeEnum` is stored sequentially. When called with
  // `ScopeIdTypeEnum::None`, returns the full count of scopes.
  auto GetScopeIdOffset(ScopeIdTypeEnum id_enum) const -> int;

  auto AddBlockLabel(
      ScopeId scope_id, InstBlockId block_id, std::string name = "",
      std::variant<LocId, uint64_t> loc_id_or_fingerprint = LocId::None)
      -> void;

  // Finds and adds a suitable block label for the given SemIR instruction that
  // represents some kind of branch.
  auto AddBlockLabel(ScopeId scope_id, LocId loc_id, AnyBranch branch) -> void;

  // Adds a scope and instructions to walk. Avoids recursion while allowing
  // the loop to below add more instructions during iteration. The new
  // instructions are pushed such that they will be the next to be walked.
  // Internally that means they are reversed and added to the end of the vector,
  // since we pop from the back of the vector.
  auto PushBlockInsts(ScopeId scope_id, llvm::ArrayRef<InstId> inst_ids)
      -> void;
  auto PushBlockId(ScopeId scope_id, InstBlockId block_id) -> void;

  // Pushes generic information for an entity.
  auto PushGeneric(ScopeId scope_id, GenericId generic_id) -> void;

  // Names an entity, and pushes processing of its blocks.
  auto PushEntity(AssociatedConstantId associated_constant_id, ScopeId scope_id,
                  Scope& scope) -> void;
  auto PushEntity(ClassId class_id, ScopeId scope_id, Scope& scope) -> void;
  auto PushEntity(FunctionId function_id, ScopeId scope_id, Scope& scope)
      -> void;
  auto PushEntity(CppOverloadSetId cpp_overload_set_id, ScopeId scope_id,
                  Scope& scope) -> void;
  auto PushEntity(ImplId impl_id, ScopeId scope_id, Scope& scope) -> void;
  auto PushEntity(InterfaceId interface_id, ScopeId scope_id, Scope& scope)
      -> void;
  auto PushEntity(NamedConstraintId named_constraint_id, ScopeId scope_id,
                  Scope& scope) -> void;
  auto PushEntity(VtableId vtable_id, ScopeId scope_id, Scope& scope) -> void;

  // Always returns the name of the entity. May push it if it has not yet been
  // pushed.
  template <typename EntityIdT>
  auto MaybePushEntity(EntityIdT entity_id) -> llvm::StringRef {
    auto scope_id = GetScopeFor(entity_id);
    auto& scope = GetScopeInfo(scope_id);
    if (!scope.name) {
      PushEntity(entity_id, scope_id, scope);
    }
    return scope.name.GetBaseName();
  }

  const File* sem_ir_;
  InstFingerprinter fingerprinter_;

  // The namespace for entity names. Names within this namespace are prefixed
  // with `@` in formatted SemIR.
  Namespace globals_;
  // The enclosing scope and name for each instruction, indexed by the InstId's
  // index.
  std::vector<std::pair<ScopeId, Namespace::Name>> insts_;
  // The enclosing scope and name for each block that might be a branch target,
  // indexed by the InstBlockId's index.
  std::vector<std::pair<ScopeId, Namespace::Name>> labels_;
  // The scopes corresponding to ScopeId values.
  std::vector<Scope> scopes_;
  // The scope IDs corresponding to generics. The vector indexes are the
  // GenericId index.
  std::vector<ScopeId> generic_scopes_;

  // The stack of instructions to name.
  llvm::SmallVector<std::pair<ScopeId, InstId>> inst_stack_;
  // The stack of blocks to traverse.
  llvm::SmallVector<std::pair<ScopeId, InstBlockId>> inst_block_stack_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_NAMER_H_
