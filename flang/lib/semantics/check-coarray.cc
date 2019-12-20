//===-- lib/semantics/check-coarray.cc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#include "check-coarray.h"
#include "expression.h"
#include "tools.h"
#include "../common/indirection.h"
#include "../evaluate/expression.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
#include "../parser/tools.h"

namespace Fortran::semantics {

template<typename T>
static void CheckTeamType(SemanticsContext &context, const T &x) {
  if (const auto *expr{GetExpr(x)}) {
    if (!IsTeamType(evaluate::GetDerivedTypeSpec(expr->GetType()))) {
      context.Say(parser::FindSourceLocation(x),  // C1114
          "Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV"_err_en_US);
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
  context_.Say(name1, std::move(msg1), name1)
      .Attach(name2, std::move(msg2), name2);
}

}
