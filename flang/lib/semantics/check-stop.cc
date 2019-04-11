// Copyright (c) 2019, Arm Ltd.  All rights reserved.
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

#include "check-stop.h"
#include "semantics.h"
#include "tools.h"
#include "../common/Fortran.h"
#include "../evaluate/expression.h"
#include "../parser/parse-tree.h"
#include <optional>

void Fortran::semantics::StopChecker::Enter(
    const Fortran::parser::StopStmt &stmt) {
  const auto &sc{std::get<std::optional<Fortran::parser::StopCode>>(stmt.t)};
  const auto &sle{
      std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(stmt.t)};

  if (sc.has_value()) {
    const Fortran::parser::CharBlock &source{sc.value().v.thing.source};
    const auto &expr{*(sc.value().v.thing.typedExpr)};

    if (!(Fortran::semantics::ExprIsScalar(expr))) {
      context_.Say(source, "Stop code must be a scalar"_err_en_US);
    } else {
      if (Fortran::semantics::ExprHasTypeCategory(
              expr, Fortran::common::TypeCategory::Integer)) {
        // C1171 default kind
        if (!(Fortran::semantics::ExprHasTypeKind(expr,
                context_.defaultKinds().GetDefaultKind(
                    Fortran::common::TypeCategory::Integer)))) {
          context_.Say(
              source, "Integer stop code must be of default kind"_err_en_US);
        }
      } else if (Fortran::semantics::ExprHasTypeCategory(
                     expr, Fortran::common::TypeCategory::Character)) {
        // R1162 spells scalar-DEFAULT-char-expr
        if (!(Fortran::semantics::ExprHasTypeKind(expr,
                context_.defaultKinds().GetDefaultKind(
                    Fortran::common::TypeCategory::Character)))) {
          context_.Say(
              source, "Character stop code must be of default kind"_err_en_US);
        }
      } else {
        context_.Say(
            source, "Stop code must be of INTEGER or CHARACTER type"_err_en_US);
      }
    }
  }
  if (sle.has_value()) {
    const Fortran::parser::CharBlock &source{
        sle.value().thing.thing.value().source};
    const auto &expr{*(sle.value().thing.thing.value().typedExpr)};

    if (!(Fortran::semantics::ExprIsScalar(expr))) {
      context_.Say(source,
          "The optional QUIET parameter value must be a scalar"_err_en_US);
    } else {
      if (!(Fortran::semantics::ExprHasTypeCategory(
              expr, Fortran::common::TypeCategory::Logical))) {
        context_.Say(source,
            "The optional QUIET parameter value must be of LOGICAL type"_err_en_US);
      }
    }
  }
}
