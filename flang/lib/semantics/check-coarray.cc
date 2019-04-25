// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "check-coarray.h"
#include "expression.h"
#include "tools.h"
#include "../common/indirection.h"
#include "../evaluate/expression.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
#include "../parser/tools.h"

namespace Fortran::semantics {

// Is this a derived type from module with this name?
static bool IsDerivedTypeFromModule(
    const DerivedTypeSpec *derived, const char *module, const char *name) {
  if (!derived) {
    return false;
  } else {
    const auto &symbol{derived->typeSymbol()};
    return symbol.name() == name && symbol.owner().IsModule() &&
        symbol.owner().name() == module;
  }
}

static bool IsTeamType(const DerivedTypeSpec *derived) {
  return IsDerivedTypeFromModule(derived, "iso_fortran_env", "team_type");
}

template<typename T>
static void CheckTeamType(SemanticsContext &context, const T &x) {
  if (const auto *expr{GetExpr(x)}) {
    if (auto type{expr->GetType()}) {
      if (!IsTeamType(type->derived)) {
        context.Say(parser::FindSourceLocation(x),  // C1114
            "Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV"_err_en_US);
      }
    }
  }
}

void CoarrayChecker::Leave(const parser::ChangeTeamStmt &x) {
  CheckNamesAreDistinct(std::get<std::list<parser::CoarrayAssociation>>(x.t));
  CheckTeamType(context_, std::get<parser::TeamValue>(x.t));
}

void CoarrayChecker::Leave(const parser::SyncTeamStmt &x) {
  CheckTeamType(context_, std::get<parser::TeamValue>(x.t));
}

void CoarrayChecker::Leave(const parser::ImageSelectorSpec &x) {
  if (const auto *team{std::get_if<parser::TeamValue>(&x.u)}) {
    CheckTeamType(context_, *team);
  }
}

void CoarrayChecker::Leave(const parser::FormTeamStmt &x) {
  CheckTeamType(context_, std::get<parser::TeamVariable>(x.t));
}

// Check that coarray names and selector names are all distinct.
void CoarrayChecker::CheckNamesAreDistinct(
    const std::list<parser::CoarrayAssociation> &list) {
  std::set<parser::CharBlock> names;
  auto getPreviousUse{
      [&](const parser::Name &name) -> const parser::CharBlock * {
        auto pair{names.insert(name.source)};
        return !pair.second ? &*pair.first : nullptr;
      }};
  for (const auto &assoc : list) {
    const auto &decl{std::get<parser::CodimensionDecl>(assoc.t)};
    const auto &selector{std::get<parser::Selector>(assoc.t)};
    const auto &declName{std::get<parser::Name>(decl.t)};
    if (context_.HasError(declName)) {
      continue;  // already reported an error about this name
    }
    if (auto *prev{getPreviousUse(declName)}) {
      Say2(declName.source,  // C1113
          "Coarray '%s' was already used as a selector or coarray in this statement"_err_en_US,
          *prev, "Previous use of '%s'"_en_US);
    }
    // ResolveNames verified the selector is a simple name
    const parser::Name *name{parser::Unwrap<parser::Name>(selector)};
    if (name) {
      if (auto *prev{getPreviousUse(*name)}) {
        Say2(name->source,  // C1113, C1115
            "Selector '%s' was already used as a selector or coarray in this statement"_err_en_US,
            *prev, "Previous use of '%s'"_en_US);
      }
    }
  }
}

void CoarrayChecker::Say2(const parser::CharBlock &name1,
    parser::MessageFixedText &&msg1, const parser::CharBlock &name2,
    parser::MessageFixedText &&msg2) {
  context_.Say(name1, std::move(msg1), name1.ToString().c_str())
      .Attach(name2, std::move(msg2), name2.ToString().c_str());
}

}
