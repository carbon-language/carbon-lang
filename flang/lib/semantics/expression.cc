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
#include "../evaluate/common.h"

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

using Result = std::optional<evaluate::GenericExpr>;

// AnalyzeHelper is a local template function that keeps the API
// member function ExpressionAnalyzer::Analyze from having to be a
// many-specialized template itself.
template<typename A> Result AnalyzeHelper(ExpressionAnalyzer &, const A &);

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr &expr) {
  return ea.Analyze(expr);
}

// Template wrappers are traversed with checking.
template<typename A>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const std::optional<A> &x) {
  if (x.has_value()) {
    return AnalyzeHelper(ea, *x);
  } else {
    return std::nullopt;
  }
}

template<typename A>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const common::Indirection<A> &p) {
  return AnalyzeHelper(ea, *p);
}

template<typename A>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Scalar<A> &tree) {
  Result result{AnalyzeHelper(ea, tree.thing)};
  if (result.has_value()) {
    if (result->Rank() > 1) {
      ea.context().messages.Say("must be scalar"_err_en_US);
      return std::nullopt;
    }
  }
  return result;
}

template<typename A>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Constant<A> &tree) {
  Result result{AnalyzeHelper(ea, tree.thing)};
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
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Integer<A> &tree) {
  Result result{AnalyzeHelper(ea, tree.thing)};
  if (result.has_value() &&
      !std::holds_alternative<evaluate::AnyKindIntegerExpr>(result->u)) {
    ea.context().messages.Say("must be integer"_err_en_US);
    return std::nullopt;
  }
  return result;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::CharLiteralConstant &x) {
  auto kind{ea.Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      ExpressionAnalyzer::KindParam{1})};
  switch (kind) {
#define CASE(k) \
  case k: \
    return {evaluate::GenericExpr{evaluate::AnyKindCharacterExpr{ \
        evaluate::CharacterExpr<k>{std::get<std::string>(x.t)}}}};
#undef CASE
  default:
    ea.context().messages.Say("unimplemented CHARACTER kind (%ju)"_err_en_US,
        static_cast<std::uintmax_t>(kind));
    return std::nullopt;
  }
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::CharLiteralConstantSubstring &x) {
  const auto &range{std::get<parser::SubstringRange>(x.t)};
  const std::optional<parser::ScalarIntExpr> &lbTree{std::get<0>(range.t)};
  const std::optional<parser::ScalarIntExpr> &ubTree{std::get<1>(range.t)};
  if (!lbTree.has_value() && !ubTree.has_value()) {
    // "..."(:)
    return AnalyzeHelper(ea, std::get<parser::CharLiteralConstant>(x.t));
  }
  // TODO: ensure that any kind parameter is 1
  std::string str{std::get<parser::CharLiteralConstant>(x.t).GetString()};
  std::optional<evaluate::SubscriptIntegerExpr> lb, ub;
  if (lbTree.has_value()) {
    if (Result lbExpr{AnalyzeHelper(ea, *lbTree)}) {
      if (auto *ie{std::get_if<evaluate::AnyKindIntegerExpr>(&lbExpr->u)}) {
        lb = evaluate::SubscriptIntegerExpr{std::move(*ie)};
      } else {
        ea.context().messages.Say(
            "scalar integer expression required for substring lower bound"_err_en_US);
      }
    }
  }
  if (ubTree.has_value()) {
    if (Result ubExpr{AnalyzeHelper(ea, *ubTree)}) {
      if (auto *ie{std::get_if<evaluate::AnyKindIntegerExpr>(&ubExpr->u)}) {
        ub = evaluate::SubscriptIntegerExpr{std::move(*ie)};
      } else {
        ea.context().messages.Say(
            "scalar integer expression required for substring upper bound"_err_en_US);
      }
    }
  }
  if (!lb.has_value() || !ub.has_value()) {
    return std::nullopt;
  }
  evaluate::Substring substring{std::move(str), std::move(lb), std::move(ub)};
  evaluate::CopyableIndirection<evaluate::Substring> ind{std::move(substring)};
  evaluate::CharacterExpr<1> chExpr{std::move(ind)};
  chExpr.Fold(ea.context());
  evaluate::AnyKindCharacterExpr akcExpr{std::move(chExpr)};
  evaluate::GenericExpr gExpr{std::move(akcExpr)};
  return {gExpr};
}

template<>
Result AnalyzeHelper(
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
    ea.context().messages.Say("unimplemented INTEGER kind (%ju)"_err_en_US,
        static_cast<std::uintmax_t>(kind));
    return std::nullopt;
  }
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::BOZLiteralConstant &x) {
  const char *p{x.v.data()};
  std::uint64_t base{16};
  switch (*p++) {
  case 'b': base = 2; break;
  case 'o': base = 8; break;
  case 'z': break;
  case 'x': break;
  default: CRASH_NO_CASE;
  }
  CHECK(*p == '"');
  auto value{evaluate::BOZLiteralConstant::ReadUnsigned(++p, base)};
  if (*p != '"') {
    ea.context().messages.Say(
        "invalid digit ('%c') in BOZ literal %s"_err_en_US, *p, x.v.data());
    return std::nullopt;
  }
  if (value.overflow) {
    ea.context().messages.Say("BOZ literal %s too large"_err_en_US, x.v.data());
    return std::nullopt;
  }
  return {evaluate::GenericExpr{value.value}};
}

template<int KIND>
Result ReadRealLiteral(
    parser::CharBlock source, evaluate::FoldingContext &context) {
  using valueType = typename evaluate::RealExpr<KIND>::Scalar;
  const char *p{source.begin()};
  auto valWithFlags{valueType::Read(p, context.rounding)};
  CHECK(p == source.end());
  evaluate::RealFlagWarnings(
      context, valWithFlags.flags, "conversion of REAL literal");
  auto value{valWithFlags.value};
  if (context.flushDenormalsToZero) {
    value = value.FlushDenormalToZero();
  }
  return {evaluate::GenericExpr{
      evaluate::AnyKindRealExpr{evaluate::RealExpr<KIND>{value}}}};
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::RealLiteralConstant &x) {
  // Use a local message context around the real literal.
  parser::ContextualMessages ctxMsgs{x.real.source, ea.context().messages};
  evaluate::FoldingContext foldingContext{ctxMsgs, ea.context()};
  // If a kind parameter appears, it takes precedence.  In the absence of
  // an explicit kind parameter, the exponent letter (e.g., 'e'/'d')
  // determines the kind.
  typename ExpressionAnalyzer::KindParam defaultKind{ea.defaultRealKind()};
  const char *end{x.real.source.end()};
  for (const char *p{x.real.source.begin()}; p < end; ++p) {
    if (parser::IsLetter(*p)) {
      switch (*p) {
      case 'e': defaultKind = 4; break;
      case 'd': defaultKind = 8; break;
      case 'q': defaultKind = 16; break;
      default: ctxMsgs.Say("unknown exponent letter '%c'"_err_en_US, *p);
      }
      break;
    }
  }
  auto kind{ea.Analyze(x.kind, defaultKind)};
  switch (kind) {
#define CASE(k) \
  case k: return ReadRealLiteral<k>(x.real.source, foldingContext);
    FOR_EACH_REAL_KIND(CASE, )
#undef CASE
  default:
    ctxMsgs.Say("unimplemented REAL kind (%ju)"_err_en_US,
        static_cast<std::uintmax_t>(kind));
    return std::nullopt;
  }
}

// Per F'2018 R718, if both components are INTEGER, they are both converted
// to default REAL and the result is default COMPLEX.  Otherwise, the
// kind of the result is the kind of largest REAL component, and the other
// component is converted if necessary its type.
template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::ComplexLiteralConstant &z) {
#if 0  // TODO pmk next
  parser::ComplexPart re{std::get<0>(z.t)}, im{std::get<1>(z.t)};
#endif
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::HollerithLiteralConstant &x) {
  evaluate::Expr<evaluate::DefaultCharacter> expr{x.v};
  return {evaluate::GenericExpr{evaluate::AnyKindCharacterExpr{expr}}};
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::LogicalLiteralConstant &x) {
  auto kind{ea.Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      ea.defaultLogicalKind())};
  bool value{std::get<bool>(x.t)};
  switch (kind) {
#define CASE(k) \
  case k: \
    return {evaluate::GenericExpr{ \
        evaluate::AnyKindLogicalExpr{evaluate::LogicalExpr<k>{value}}}};
    FOR_EACH_LOGICAL_KIND(CASE, )
#undef CASE
  default:
    ea.context().messages.Say("unimplemented LOGICAL kind (%ju)"_err_en_US,
        static_cast<std::uintmax_t>(kind));
    return std::nullopt;
  }
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::LiteralConstant &x) {
  return std::visit([&](const auto &c) { return AnalyzeHelper(ea, c); }, x.u);
}

template<> Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Name &n) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::ArrayConstructor &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::StructureConstructor &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::TypeParamInquiry &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::FunctionReference &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Expr::Parentheses &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::UnaryPlus &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Negate &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::NOT &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Expr::PercentLoc &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Expr::DefinedUnary &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Power &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Multiply &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Divide &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Add &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Subtract &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::Concat &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::LT &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::LE &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::EQ &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::NE &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::GE &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::GT &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::AND &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::OR &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::EQV &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::NEQV &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(ExpressionAnalyzer &ea, const parser::Expr::XOR &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Expr::DefinedBinary &x) {
  // TODO
  return std::nullopt;
}

template<>
Result AnalyzeHelper(
    ExpressionAnalyzer &ea, const parser::Expr::ComplexConstructor &x) {
  // TODO
  return std::nullopt;
}

Result ExpressionAnalyzer::Analyze(const parser::Expr &x) {
  return std::visit(common::visitors{[&](const parser::LiteralConstant &c) {
                                       return AnalyzeHelper(*this, c);
                                     },
                        // TODO: remaining cases
                        [&](const auto &) { return Result{}; }},
      x.u);
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
            if (Result oge{AnalyzeHelper(*this, n)}) {
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

}  // namespace Fortran::semantics
