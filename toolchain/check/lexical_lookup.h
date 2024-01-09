
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_LEXICAL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_LEXICAL_LOOKUP_H_

#include "toolchain/check/scope_index.h"
#include "toolchain/sem_ir/file.h"
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

  explicit LexicalLookup(StringStoreWrapper<IdentifierId>& identifiers)
      : identifiers_(&identifiers), lookup_(identifiers_->size() + Offset) {}

  ~LexicalLookup() {
    CARBON_CHECK(lookup_.size() == identifiers_->size() + Offset)
        << lookup_.size() << " must match " << identifiers_->size() << " + "
        << SemIR::NameId::SpecialValueCount
        << " + 1 (for Invalid); something may have been added incorrectly";
  }

  // Handles both adding the identifier and resizing lookup_ to accommodate the
  // new entry. `identifiers().Add` must not be called directly once checking
  // has begun.
  auto AddIdentifier(llvm::StringRef name) -> IdentifierId {
    auto id = identifiers_->Add(name);
    // Bear in mind that Add was not guaranteed to actually change the size.
    lookup_.resize(identifiers_->size() + Offset);
    return id;
  }

  // Returns the lexical lookup results for a name.
  auto Get(SemIR::NameId name_id) -> llvm::SmallVector<Result>& {
    return lookup_[name_id.index + Offset];
  }

 private:
  // The offset used when accessing lookup_ with NameId::index.
  static constexpr int Offset = SemIR::NameId::SpecialValueCount + 1;

  StringStoreWrapper<IdentifierId>* identifiers_;

  // Maps identifiers to name lookup results.
  // TODO: Consider TinyPtrVector<Result> or similar.
  llvm::SmallVector<llvm::SmallVector<Result>> lookup_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_LEXICAL_LOOKUP_H_
