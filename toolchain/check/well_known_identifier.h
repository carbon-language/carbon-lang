// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_WELL_KNOWN_IDENTIFIER_H_
#define CARBON_TOOLCHAIN_CHECK_WELL_KNOWN_IDENTIFIER_H_

#include "common/enum_base.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

CARBON_DEFINE_RAW_ENUM_CLASS(WellKnownIdentifier, uint8_t) {
#define CARBON_WELL_KNOWN_IDENTIFIER(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/check/well_known_identifier.def"
};

// A well-known identifier that's part of the language, but not a builtin or
// keyword.
//
// For example, `AddWith` is a well-known identifier because `Core.AddWith` is
// part of the language design. The `AddWith` type is found through name lookup,
// and so we need to know its identifier in order to desugar `+`.
class WellKnownIdentifier : public CARBON_ENUM_BASE(WellKnownIdentifier) {
 public:
#define CARBON_WELL_KNOWN_IDENTIFIER(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/check/well_known_identifier.def"

 private:
  // Exposes `AsInt`.
  friend class WellKnownIdentifierCache;
};

#define CARBON_WELL_KNOWN_IDENTIFIER(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(WellKnownIdentifier, Name)
#include "toolchain/check/well_known_identifier.def"

// A cache of added well-known identifiers. These are added to the identifier
// store on first use.
class WellKnownIdentifierCache {
 public:
  explicit WellKnownIdentifierCache(
      SharedValueStores::IdentifierStore* identifiers)
      : identifiers_(identifiers) {}

  // Returns the `NameId` for a `WellKnownIdentifier`.
  auto AddNameId(WellKnownIdentifier identifier) -> SemIR::NameId {
    auto& value = cache_[identifier.AsInt()];
    if (!value.has_value()) {
      value =
          SemIR::NameId::ForIdentifier(identifiers_->Add(identifier.name()));
    }
    return value;
  }

 private:
  // The number of cache entries.
  static constexpr int CacheSize = 0
#define CARBON_WELL_KNOWN_IDENTIFIER(Name) +1
#include "toolchain/check/well_known_identifier.def"
      ;

  // A pointer for adding identifiers.
  SharedValueStores::IdentifierStore* identifiers_;

  // The cache of added identifiers. These are stored as a `NameId` because the
  // `IdentifierId` isn't directly used.
  SemIR::NameId cache_[CacheSize] = {
#define CARBON_WELL_KNOWN_IDENTIFIER(Name) SemIR::NameId::None,
#include "toolchain/check/well_known_identifier.def"
  };
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_WELL_KNOWN_IDENTIFIER_H_
