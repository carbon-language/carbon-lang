// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CORE_IDENTIFIER_H_
#define CARBON_TOOLCHAIN_CHECK_CORE_IDENTIFIER_H_

#include "common/enum_base.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

CARBON_DEFINE_RAW_ENUM_CLASS(CoreIdentifier, uint8_t) {
#define CARBON_CORE_IDENTIFIER(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/check/core_identifier.def"
};

// An identifier in `Core` that's significant for the language (typically used
// by desugaring) but not a builtin or keyword. Note this includes non-top-level
// identifiers, such as both `AddWith` and `Op` in `Core.AddWith.Op`.
class CoreIdentifier : public CARBON_ENUM_BASE(CoreIdentifier) {
 public:
#define CARBON_CORE_IDENTIFIER(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/check/core_identifier.def"

 private:
  // Exposes `AsInt`.
  friend class CoreIdentifierCache;
};

#define CARBON_CORE_IDENTIFIER(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(CoreIdentifier, Name)
#include "toolchain/check/core_identifier.def"

// A cache of added `Core` identifiers. These are added to the identifier
// store on first use.
class CoreIdentifierCache {
 public:
  explicit CoreIdentifierCache(SharedValueStores::IdentifierStore* identifiers)
      : identifiers_(identifiers) {}

  // Returns the `NameId` for a `CoreIdentifier`.
  auto AddNameId(CoreIdentifier identifier) -> SemIR::NameId {
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
#define CARBON_CORE_IDENTIFIER(Name) +1
#include "toolchain/check/core_identifier.def"
      ;

  // A pointer for adding identifiers.
  SharedValueStores::IdentifierStore* identifiers_;

  // The cache of added identifiers. These are stored as a `NameId` because the
  // `IdentifierId` isn't directly used.
  SemIR::NameId cache_[CacheSize] = {
#define CARBON_CORE_IDENTIFIER(Name) SemIR::NameId::None,
#include "toolchain/check/core_identifier.def"
  };
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CORE_IDENTIFIER_H_
