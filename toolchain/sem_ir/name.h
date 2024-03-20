// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NAME_H_
#define CARBON_TOOLCHAIN_SEM_IR_NAME_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

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

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NAME_H_
