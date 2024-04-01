
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_LEXICAL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_LEXICAL_LOOKUP_H_

#include "toolchain/check/scope_index.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Manages lexical lookup information for NameIds.
//
// Values are a stack of name lookup results in the ancestor scopes. This offers
// constant-time lookup of names, regardless of how many scopes exist between
// the name declaration and reference. The corresponding scope for each lookup
// result is tracked, so that lexical lookup results can be interleaved with
// lookup results from non-lexical scopes such as classes.
class LexicalLookup {
 public:
  // A lookup result.
  struct Result {
    // The instruction that was added to lookup.
    SemIR::InstId inst_id;
    // The scope in which the instruction was added.
    ScopeIndex scope_index;
  };

  // A lookup result that has been temporarily removed from scope.
  struct SuspendedResult {
    // The lookup index. This is notionally a size_t, but is stored in 32 bits
    // to keep this type small, which helps to keep SuspendedFunctions small.
    uint32_t index;
    // The lookup result.
    SemIR::InstId inst_id;
  };

  explicit LexicalLookup(const StringStoreWrapper<IdentifierId>& identifiers)
      : lookup_(identifiers.size() + SemIR::NameId::NonIndexValueCount) {}

  // Returns the lexical lookup results for a name.
  auto Get(SemIR::NameId name_id) -> llvm::SmallVector<Result, 2>& {
    auto index = GetLookupIndex(name_id);
    CARBON_CHECK(index < lookup_.size())
        << "An identifier was added after the Context was initialized. "
           "Currently, we expect that new identifiers will never be used with "
           "lexical lookup (they're added for things like detecting name "
           "collisions in imports). That might change with metaprogramming: if "
           "it does, we may need to start resizing `lookup_`, either on each "
           "identifier addition or in Get` where this CHECK currently fires.";
    return lookup_[index];
  }

  // Temporarily remove the top lookup result for `name_id` from scope.
  auto Suspend(SemIR::NameId name_id) -> SuspendedResult {
    auto index = GetLookupIndex(name_id);
    auto& results = lookup_[index];
    CARBON_CHECK(!results.empty())
        << "Suspending a nonexistent result for " << name_id << ".";
    CARBON_CHECK(index <= std::numeric_limits<uint32_t>::max())
        << "Unexpectedly large index " << index << " for name ID";
    return {static_cast<uint32_t>(index), results.pop_back_val().inst_id};
  }

  // Restore a previously-suspended lookup result.
  auto Restore(SuspendedResult sus, ScopeIndex index) -> void {
    lookup_[sus.index].push_back({sus.inst_id, index});
  }

 private:
  // Get the index at which the specified name is stored in `lookup_`.
  auto GetLookupIndex(SemIR::NameId name_id) -> size_t {
    return static_cast<ssize_t>(name_id.index) +
           SemIR::NameId::NonIndexValueCount;
  }

  // Maps identifiers to name lookup results.
  // TODO: Consider TinyPtrVector<Result> or similar. For now, use a small size
  // of 2 to cover the common case.
  llvm::SmallVector<llvm::SmallVector<Result, 2>> lookup_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_LEXICAL_LOOKUP_H_
