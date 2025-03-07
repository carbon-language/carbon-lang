// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/name.h"

#include "llvm/ADT/StringSwitch.h"

namespace Carbon::SemIR {

// Get the spelling to use for a special name.
static auto GetSpecialName(NameId name_id, bool for_ir) -> llvm::StringRef {
  if (name_id == NameId::None) {
    return for_ir ? "" : "<none>";
  }

  auto special_name_id = name_id.AsSpecialNameId();
  CARBON_CHECK(special_name_id, "Not a special name");
  switch (*special_name_id) {
    case NameId::SpecialNameId::Base:
      return "base";
    case NameId::SpecialNameId::ChoiceDiscriminant:
      return "discriminant";
    case NameId::SpecialNameId::Core:
      return "Core";
    case NameId::SpecialNameId::Destroy:
      return "destroy";
    case NameId::SpecialNameId::PackageNamespace:
      return "package";
    case NameId::SpecialNameId::PeriodSelf:
      return ".Self";
    case NameId::SpecialNameId::ReturnSlot:
      return for_ir ? "return" : "<return slot>";
    case NameId::SpecialNameId::SelfType:
      return "Self";
    case NameId::SpecialNameId::SelfValue:
      return "self";
    case NameId::SpecialNameId::Vptr:
      return for_ir ? "vptr" : "<vptr>";
  }
}

auto NameStoreWrapper::GetFormatted(NameId name_id) const -> llvm::StringRef {
  // If the name is an identifier name with a keyword spelling, format it with
  // an `r#` prefix. Format any other identifier name as just the identifier.
  if (auto string_name = GetAsStringIfIdentifier(name_id)) {
    return llvm::StringSwitch<llvm::StringRef>(*string_name)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, "r#" Spelling)
#include "toolchain/lex/token_kind.def"
        .Default(*string_name);
  }
  return GetSpecialName(name_id, /*for_ir=*/false);
}

auto NameStoreWrapper::GetIRBaseName(NameId name_id) const -> llvm::StringRef {
  if (auto string_name = GetAsStringIfIdentifier(name_id)) {
    return *string_name;
  }
  return GetSpecialName(name_id, /*for_ir=*/true);
}

}  // namespace Carbon::SemIR
