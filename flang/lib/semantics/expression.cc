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

template<typename A>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const A &tree) {
  return ea.Analyze(tree);
}

template<typename A>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Scalar<A> &tree) {
  std::optional<evaluate::GenericExpr> result{AnalyzeHelper(ea, tree.thing)};
  if (result.has_value()) {
    if (result->Rank() > 1) {
      ea.context().messages.Say("must be scalar"_err_en_US);
      return std::nullopt;
    }
  }
  return result;
}

template<typename A>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Constant<A> &tree) {
  std::optional<evaluate::GenericExpr> result{AnalyzeHelper(ea, tree.thing)};
  if (result.has_value()) {
    result->Fold(ea.context());
    if (!result->ScalarValue().has_value()) {
      ea.context().messages.Say("must be constant"_err_en_US);
      return std::nullopt;
    }
  }
  return result;
}

template<typename A>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Integer<A> &tree) {
  std::optional<evaluate::GenericExpr> result{AnalyzeHelper(ea, tree.thing)};
  if (result.has_value() &&
      !std::holds_alternative<evaluate::AnyKindIntegerExpr>(result->u)) {
    ea.context().messages.Say("must be integer"_err_en_US);
    return std::nullopt;
  }
  return result;
}

template<>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Name &n) {
  // TODO
  return std::nullopt;
}

ExpressionAnalyzer::KindParam ExpressionAnalyzer::Analyze(
    const std::optional<parser::KindParam> &kindParam, KindParam defaultKind,
    KindParam kanjiKind) {
  if (!kindParam.has_value()) {
    return defaultKind;
  }
  return std::visit(
      common::visitors{
          [](std::uint64_t k) { return static_cast<KindParam>(k); },
          [&](const parser::Scalar<
              parser::Integer<parser::Constant<parser::Name>>> &n) {
            if (std::optional<evaluate::GenericExpr> oge{
                    AnalyzeHelper(*this, n)}) {
              if (std::optional<evaluate::GenericScalar> ogs{
                      oge->ScalarValue()}) {
                // TODO pmk more here next
              }
            }
            return defaultKind;
          },
          [&](parser::KindParam::Kanji) {
            if (kanjiKind >= 0) {
              return kanjiKind;
            }
            context().messages.Say("Kanji not allowed here"_err_en_US);
            return defaultKind;
          }},
      kindParam->u);
}

template<>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::IntLiteralConstant &x) {
  auto kind{ea.Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      ea.defaultIntegerKind())};
  std::uint64_t value{std::get<std::uint64_t>(x.t)};
  switch (kind) {
#define CASE(k) \
  case k: \
    return {evaluate::GenericExpr{ \
        evaluate::AnyKindIntegerExpr{evaluate::IntegerExpr<k>{value}}}};
    FOR_EACH_INTEGER_KIND(CASE, )
#undef CASE
  default:
    ea.context().messages.Say(parser::MessageFormattedText{
        "unimplemented INTEGER kind (%ju)"_err_en_US,
        static_cast<std::uintmax_t>(kind)});
    return std::nullopt;
  }
}

template<>
std::optional<evaluate::GenericExpr> AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::LiteralConstant &x) {
  return std::visit(
      common::visitors{[&](const parser::IntLiteralConstant &c) {
                         return AnalyzeHelper(ea, c);
                       },
          // TODO next [&](const parser::RealLiteralConstant &c) { return
          // AnalyzeHelper(ea, c); },
          // TODO: remaining cases
          [&](const auto &) { return std::optional<evaluate::GenericExpr>{}; }},
      x.u);
}

std::optional<evaluate::GenericExpr> ExpressionAnalyzer::Analyze(
    const parser::Expr &x) {
  return std::visit(
      common::visitors{[&](const parser::LiteralConstant &c) {
                         return AnalyzeHelper(*this, c);
                       },
          // TODO: remaining cases
          [&](const auto &) { return std::optional<evaluate::GenericExpr>{}; }},
      x.u);
}
}  // namespace Fortran::semantics
