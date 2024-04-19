// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_NAMER_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_NAMER_H_

#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

// Assigns names to instructions, blocks, and scopes in the Semantics IR.
class InstNamer {
 public:
  // int32_t matches the input value size.
  // NOLINTNEXTLINE(performance-enum-size)
  enum class ScopeId : int32_t {
    None = -1,
    File = 0,
    ImportRef = 1,
    Constants = 2,
    FirstFunction = 3,
  };
  static_assert(sizeof(ScopeId) == sizeof(FunctionId));

  struct NumberOfScopesTag {};

  // Construct the instruction namer, and assign names to all instructions in
  // the provided file.
  InstNamer(const Lex::TokenizedBuffer& tokenized_buffer,
            const Parse::Tree& parse_tree, const File& sem_ir);

  // Returns the scope ID corresponding to an ID of a function, class, or
  // interface.
  template <typename IdT>
  auto GetScopeFor(IdT id) const -> ScopeId {
    auto index = static_cast<int32_t>(ScopeId::FirstFunction);

    if constexpr (!std::same_as<FunctionId, IdT>) {
      index += sem_ir_.functions().size();
      if constexpr (!std::same_as<ClassId, IdT>) {
        index += sem_ir_.classes().size();
        if constexpr (!std::same_as<InterfaceId, IdT>) {
          index += sem_ir_.interfaces().size();
          if constexpr (!std::same_as<ImplId, IdT>) {
            index += sem_ir_.impls().size();
            static_assert(std::same_as<NumberOfScopesTag, IdT>,
                          "Unknown ID kind for scope");
          }
        }
      }
    }
    if constexpr (!std::same_as<NumberOfScopesTag, IdT>) {
      index += id.index;
    }
    return static_cast<ScopeId>(index);
  }

  // Returns the IR name for the specified scope.
  auto GetScopeName(ScopeId scope) const -> std::string;

  // Returns the IR name to use for a function, class, or interface.
  template <typename IdT>
  auto GetNameFor(IdT id) const -> std::string {
    if (!id.is_valid()) {
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

 private:
  // A space in which unique names can be allocated.
  struct Namespace {
    // A result of a name lookup.
    struct NameResult;

    // A name in a namespace, which might be redirected to refer to another name
    // for disambiguation purposes.
    class Name {
     public:
      Name() : value_(nullptr) {}
      explicit Name(llvm::StringMapIterator<NameResult> it) : value_(&*it) {}

      explicit operator bool() const { return value_; }

      auto str() const -> llvm::StringRef;

      auto SetFallback(Name name) -> void { value_->second.fallback = name; }

      auto SetAmbiguous() -> void { value_->second.ambiguous = true; }

     private:
      llvm::StringMapEntry<NameResult>* value_ = nullptr;
    };

    struct NameResult {
      bool ambiguous = false;
      Name fallback = Name();
    };

    llvm::StringMap<NameResult> allocated = {};
    int unnamed_count = 0;

    auto AddNameUnchecked(llvm::StringRef name) -> Name {
      return Name(allocated.insert({name, NameResult()}).first);
    }

    auto AllocateName(const InstNamer& inst_namer, SemIR::LocId loc_id,
                      std::string name) -> Name;
  };

  // A named scope that contains named entities.
  struct Scope {
    Namespace::Name name;
    Namespace insts;
    Namespace labels;
  };

  auto GetScopeInfo(ScopeId scope_id) -> Scope& {
    return scopes[static_cast<int>(scope_id)];
  }

  auto GetScopeInfo(ScopeId scope_id) const -> const Scope& {
    return scopes[static_cast<int>(scope_id)];
  }

  auto AddBlockLabel(ScopeId scope_id, InstBlockId block_id,
                     std::string name = "",
                     SemIR::LocId loc_id = SemIR::LocId::Invalid) -> void;

  // Finds and adds a suitable block label for the given SemIR instruction that
  // represents some kind of branch.
  auto AddBlockLabel(ScopeId scope_id, SemIR::LocId loc_id, AnyBranch branch)
      -> void;

  auto CollectNamesInBlock(ScopeId scope_id, InstBlockId block_id) -> void;

  auto CollectNamesInBlock(ScopeId scope_id, llvm::ArrayRef<InstId> block)
      -> void;

  const Lex::TokenizedBuffer& tokenized_buffer_;
  const Parse::Tree& parse_tree_;
  const File& sem_ir_;

  Namespace globals;
  std::vector<std::pair<ScopeId, Namespace::Name>> insts;
  std::vector<std::pair<ScopeId, Namespace::Name>> labels;
  std::vector<Scope> scopes;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_NAMER_H_
