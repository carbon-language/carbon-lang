// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "expression.h"
#include "../common/idioms.h"

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

std::optional<evaluate::GenericExpr> ExpressionAnalyzer::Analyze(
    const parser::Expr &x) {
  return std::visit(
      common::visitors{
          [&](const parser::LiteralConstant &c) { return Analyze(c); },
          [&](const auto &) { return std::optional<evaluate::GenericExpr>{}; }},
      x.u);
}

std::optional<evaluate::GenericExpr> ExpressionAnalyzer::Analyze(
    const parser::IntLiteralConstant &x) {
  std::uint64_t kind = defaultIntegerKind_;
  const auto &kindParam{std::get<std::optional<parser::KindParam>>(x.t)};
  if (kindParam.has_value()) {
    std::visit(common::visitors{[&](std::uint64_t k) { kind = k; },
                   [&](const auto &) {
                     messages_.Say(at_, "unimp kind param"_err_en_US);
                   }},
        kindParam->u);
  }
  std::uint64_t value{std::get<std::uint64_t>(x.t)};
  switch (kind) {
  case 4:
    return {evaluate::GenericExpr{
        evaluate::GenericIntegerExpr{evaluate::IntegerExpr<4>{value}}}};
  default:
    messages_.Say(at_,
        parser::MessageFormattedText{
            "unimplemented INTEGER kind (%ju)"_err_en_US,
            static_cast<std::uintmax_t>(kind)});
    return {};
  }
}

std::optional<evaluate::GenericExpr> ExpressionAnalyzer::Analyze(
    const parser::LiteralConstant &x) {
  return std::visit(
      common::visitors{
          [&](const parser::IntLiteralConstant &c) { return Analyze(c); },
          [&](const auto &) { return std::optional<evaluate::GenericExpr>{}; }},
      x.u);
}
}  // namespace Fortran::semantics
