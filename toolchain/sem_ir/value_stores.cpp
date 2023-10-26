// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/value_stores.h"

#include "llvm/ADT/StringSwitch.h"

namespace Carbon::SemIR {

// Get the spelling to use for a special name.
static auto GetSpecialName(NameId name_id, bool for_ir) -> llvm::StringRef {
  switch (name_id.index) {
    case NameId::Invalid.index:
      return for_ir ? "" : "<invalid>";
    case NameId::SelfValue.index:
      return "self";
    case NameId::ReturnSlot.index:
      return for_ir ? "return" : "<return slot>";
    default:
      CARBON_FATAL() << "Unknown special name";
  }
}

auto NameStore::GetFormatted(NameId name_id) const -> llvm::StringRef {
  // If the name is a string name with a keyword spelling, format it with an
  // `r#` prefix. Return any other string name as-is.
  if (auto string_name = GetAsString(name_id)) {
    return llvm::StringSwitch<llvm::StringRef>(*string_name)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, "r#" Spelling)
#include "toolchain/lex/token_kind.def"
        .Default(*string_name);
  }
  return GetSpecialName(name_id, /*for_ir=*/false);
}

auto NameStore::GetIRBaseName(NameId name_id) const -> llvm::StringRef {
  if (auto string_name = GetAsString(name_id)) {
    return *string_name;
  }
  return GetSpecialName(name_id, /*for_ir=*/true);
}

}  // namespace Carbon::SemIR
