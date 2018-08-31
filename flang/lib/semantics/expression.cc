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
#include "symbol.h"
#include "../common/idioms.h"
#include "../evaluate/common.h"
#include "../evaluate/tools.h"
#include <functional>
#include <optional>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

using common::TypeCategory;

using MaybeExpr = std::optional<Expr<SomeType>>;

template<typename A> MaybeExpr AsMaybeExpr(std::optional<A> &&x) {
  if (x.has_value()) {
    return {AsGenericExpr(AsCategoryExpr(AsExpr(std::move(*x))))};
  }
  return std::nullopt;
}

template<TypeCategory CAT, int KIND>
MaybeExpr PackageGeneric(std::optional<Expr<Type<CAT, KIND>>> &&x) {
  if (x.has_value()) {
    return {AsGenericExpr(AsCategoryExpr(std::move(*x)))};
  }
  return std::nullopt;
}

template<TypeCategory CAT>
MaybeExpr AsMaybeExpr(std::optional<Expr<SomeKind<CAT>>> &&x) {
  if (x.has_value()) {
    return {AsGenericExpr(std::move(*x))};
  }
  return std::nullopt;
}

// This local class wraps some state and a highly overloaded
// Analyze() member function that converts parse trees into generic
// expressions.
struct ExprAnalyzer {
  using MaybeIntExpr = std::optional<Expr<SomeInteger>>;

  ExprAnalyzer(
      FoldingContext &ctx, const semantics::IntrinsicTypeDefaultKinds &dfts)
    : context{ctx}, defaults{dfts} {}

  int Analyze(
      const std::optional<parser::KindParam> &, int defaultKind, int kanjiKind);
  MaybeExpr Analyze(const parser::Expr &);
  MaybeExpr Analyze(const parser::LiteralConstant &);
  MaybeExpr Analyze(const parser::HollerithLiteralConstant &);
  MaybeExpr Analyze(const parser::IntLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedIntLiteralConstant &);
  MaybeExpr Analyze(const parser::RealLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedRealLiteralConstant &);
  MaybeExpr Analyze(const parser::ComplexLiteralConstant &);
  MaybeExpr Analyze(const parser::BOZLiteralConstant &);
  MaybeExpr Analyze(const parser::CharLiteralConstant &);
  MaybeExpr Analyze(const parser::LogicalLiteralConstant &);
  MaybeExpr Analyze(const parser::Name &);
  MaybeExpr Analyze(const parser::NamedConstant &);
  MaybeExpr Analyze(const parser::ComplexPart &);

  MaybeExpr Analyze(const parser::Expr::Parentheses &);
  MaybeExpr Analyze(const parser::Expr::UnaryPlus &);  // TODO
  MaybeExpr Analyze(const parser::Expr::Negate &);  // TODO
  MaybeExpr Analyze(const parser::Expr::NOT &);  // TODO
  MaybeExpr Analyze(const parser::Expr::DefinedUnary &);  // TODO
  MaybeExpr Analyze(const parser::Expr::Power &);  // TODO
  MaybeExpr Analyze(const parser::Expr::Multiply &);
  MaybeExpr Analyze(const parser::Expr::Divide &);
  MaybeExpr Analyze(const parser::Expr::Add &);
  MaybeExpr Analyze(const parser::Expr::Subtract &);
  MaybeExpr Analyze(const parser::Expr::Concat &);  // TODO
  MaybeExpr Analyze(const parser::Expr::LT &);  // TODO
  MaybeExpr Analyze(const parser::Expr::LE &);  // TODO
  MaybeExpr Analyze(const parser::Expr::EQ &);  // TODO
  MaybeExpr Analyze(const parser::Expr::NE &);  // TODO
  MaybeExpr Analyze(const parser::Expr::GE &);  // TODO
  MaybeExpr Analyze(const parser::Expr::GT &);  // TODO
  MaybeExpr Analyze(const parser::Expr::AND &);  // TODO
  MaybeExpr Analyze(const parser::Expr::OR &);  // TODO
  MaybeExpr Analyze(const parser::Expr::EQV &);  // TODO
  MaybeExpr Analyze(const parser::Expr::NEQV &);  // TODO
  MaybeExpr Analyze(const parser::Expr::XOR &);  // TODO
  MaybeExpr Analyze(const parser::Expr::ComplexConstructor &);
  MaybeExpr Analyze(const parser::Expr::DefinedBinary &);  // TODO
  // TODO more remain

  std::optional<Expr<SomeComplex>> ConstructComplex(MaybeExpr &&, MaybeExpr &&);

  FoldingContext &context;
  const semantics::IntrinsicTypeDefaultKinds &defaults;
};

// This helper template function handles the Scalar<>, Integer<>, and
// Constant<> wrappers in the parse tree.
// C++ doesn't allow template specialization in a class, so this helper
// template function must be outside ExprAnalyzer and reflect back into it.
template<typename A> MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const A &x) {
  return ea.Analyze(x);
}

template<typename A>
MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const parser::Scalar<A> &x) {
  // TODO: check rank == 0
  return AnalyzeHelper(ea, x.thing);
}

template<typename A>
MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const parser::Integer<A> &x) {
  if (auto result{AnalyzeHelper(ea, x.thing)}) {
    if (std::holds_alternative<Expr<SomeInteger>>(result->u)) {
      return result;
    }
    ea.context.messages.Say("expression must be INTEGER"_err_en_US);
  }
  return std::nullopt;
}

template<typename A>
MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const parser::Constant<A> &x) {
  if (MaybeExpr result{AnalyzeHelper(ea, x.thing)}) {
    if (std::optional<Constant<SomeType>> folded{result->Fold(ea.context)}) {
      return {AsGenericExpr(std::move(*folded))};
    }
    ea.context.messages.Say("expression must be constant"_err_en_US);
  }
  return std::nullopt;
}

template<typename... As>
MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const std::variant<As...> &u) {
  return std::visit([&](const auto &x) { return AnalyzeHelper(ea, x); }, u);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr &expr) {
  return std::visit(common::visitors{[&](const parser::LiteralConstant &c) {
                                       return AnalyzeHelper(*this, c);
                                     },
                        // TODO: remaining cases
                        [&](const auto &) { return MaybeExpr{}; }},
      expr.u);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::LiteralConstant &x) {
  return std::visit([&](const auto &c) { return Analyze(c); }, x.u);
}

int ExprAnalyzer::Analyze(const std::optional<parser::KindParam> &kindParam,
    int defaultKind, int kanjiKind = -1) {
  if (!kindParam.has_value()) {
    return defaultKind;
  }
  return std::visit(
      common::visitors{[](std::uint64_t k) { return static_cast<int>(k); },
          [&](const parser::Scalar<
              parser::Integer<parser::Constant<parser::Name>>> &n) {
            if (MaybeExpr ie{AnalyzeHelper(*this, n)}) {
              if (std::optional<GenericScalar> sv{ie->ScalarValue()}) {
                if (std::optional<std::int64_t> i64{sv->ToInt64()}) {
                  std::int64_t i64v{*i64};
                  int iv = i64v;
                  if (iv == i64v) {
                    return iv;
                  }
                }
              }
            }
            context.messages.Say(
                "KIND type parameter must be a scalar integer constant"_err_en_US);
            return defaultKind;
          },
          [&](parser::KindParam::Kanji) {
            if (kanjiKind >= 0) {
              return kanjiKind;
            }
            context.messages.Say("Kanji not allowed here"_err_en_US);
            return defaultKind;
          }},
      kindParam->u);
}

// A helper class used with common::SearchDynamicTypes when constructing
// a literal constant with a dynamic kind in some type category.
template<TypeCategory CAT, typename VALUE> struct ConstantTypeVisitor {
  using Result = std::optional<Expr<SomeKind<CAT>>>;
  static constexpr std::size_t Types{std::tuple_size_v<CategoryTypes<CAT>>};

  ConstantTypeVisitor(int k, const VALUE &x) : kind{k}, value{x} {}

  template<std::size_t J> Result Test() {
    using Ty = std::tuple_element_t<J, CategoryTypes<CAT>>;
    if (kind == Ty::kind) {
      return {AsCategoryExpr(AsExpr(Constant<Ty>{std::move(value)}))};
    }
    return std::nullopt;
  }

  int kind;
  VALUE value;
};

MaybeExpr ExprAnalyzer::Analyze(const parser::HollerithLiteralConstant &x) {
  return AsMaybeExpr(common::SearchDynamicTypes(
      ConstantTypeVisitor<TypeCategory::Character, std::string>{
          defaults.defaultCharacterKind, x.v}));
}

// Common handling of parser::IntLiteralConstant and SignedIntLiteralConstant
template<typename PARSED>
MaybeExpr IntLiteralConstant(ExprAnalyzer &ea, const PARSED &x) {
  int kind{ea.Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      ea.defaults.defaultIntegerKind)};
  auto value{std::get<0>(x.t)};  // std::(u)int64_t
  auto result{common::SearchDynamicTypes(
      ConstantTypeVisitor<TypeCategory::Integer, std::int64_t>{
          kind, static_cast<std::int64_t>(value)})};
  if (!result.has_value()) {
    ea.context.messages.Say("unsupported INTEGER(KIND=%u)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::IntLiteralConstant &x) {
  return IntLiteralConstant(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::SignedIntLiteralConstant &x) {
  return IntLiteralConstant(*this, x);
}

template<typename TYPE>
Constant<TYPE> ReadRealLiteral(
    parser::CharBlock source, FoldingContext &context) {
  const char *p{source.begin()};
  auto valWithFlags{Scalar<TYPE>::Read(p, context.rounding)};
  CHECK(p == source.end());
  RealFlagWarnings(context, valWithFlags.flags, "conversion of REAL literal");
  auto value{valWithFlags.value};
  if (context.flushDenormalsToZero) {
    value = value.FlushDenormalToZero();
  }
  return {value};
}

// TODO: can this definition appear in the function belowe?
struct RealTypeVisitor {
  using Result = std::optional<Expr<SomeReal>>;
  static constexpr std::size_t Types{std::tuple_size_v<RealTypes>};

  RealTypeVisitor(int k, parser::CharBlock lit, FoldingContext &ctx)
    : kind{k}, literal{lit}, context{ctx} {}

  template<std::size_t J> Result Test() {
    using Ty = std::tuple_element_t<J, RealTypes>;
    if (kind == Ty::kind) {
      return {AsCategoryExpr(AsExpr(ReadRealLiteral<Ty>(literal, context)))};
    }
    return std::nullopt;
  }

  int kind;
  parser::CharBlock literal;
  FoldingContext &context;
};

MaybeExpr ExprAnalyzer::Analyze(const parser::RealLiteralConstant &x) {
  // Use a local message context around the real literal for better
  // provenance on any messages.
  parser::ContextualMessages ctxMsgs{x.real.source, context.messages};
  FoldingContext localFoldingContext{ctxMsgs, context};
  // If a kind parameter appears, it defines the kind of the literal and any
  // letter used in an exponent part (e.g., the 'E' in "6.02214E+23")
  // should agree.  In the absence of an explicit kind parameter, any exponent
  // letter determines the kind.  Otherwise, defaults apply.
  int defaultKind{defaults.defaultRealKind};
  const char *end{x.real.source.end()};
  std::optional<int> letterKind;
  for (const char *p{x.real.source.begin()}; p < end; ++p) {
    if (parser::IsLetter(*p)) {
      switch (*p) {
      case 'e': letterKind = 4; break;
      case 'd': letterKind = 8; break;
      case 'q': letterKind = 16; break;
      default: ctxMsgs.Say("unknown exponent letter '%c'"_err_en_US, *p);
      }
      break;
    }
  }
  if (letterKind.has_value()) {
    defaultKind = *letterKind;
  }
  auto kind{Analyze(x.kind, defaultKind)};
  if (letterKind.has_value() && kind != *letterKind) {
    ctxMsgs.Say(
        "explicit kind parameter on real constant disagrees with exponent letter"_en_US);
  }
  auto result{common::SearchDynamicTypes(
      RealTypeVisitor{kind, x.real.source, context})};
  if (!result.has_value()) {
    ctxMsgs.Say("unsupported REAL(KIND=%u)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::SignedRealLiteralConstant &x) {
  if (MaybeExpr result{Analyze(std::get<parser::RealLiteralConstant>(x.t))}) {
    if (auto sign{std::get<std::optional<parser::Sign>>(x.t)}) {
      if (sign == parser::Sign::Negative) {
        return {AsGenericExpr(-*common::GetIf<Expr<SomeReal>>(result->u))};
      }
    }
    return result;
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ComplexPart &x) {
  return AnalyzeHelper(*this, x.u);
}

// Per F'2018 R718, if both components are INTEGER, they are both converted
// to default REAL and the result is default COMPLEX.  Otherwise, the
// kind of the result is the kind of most precise REAL component, and the other
// component is converted if necessary to its type.
std::optional<Expr<SomeComplex>> ExprAnalyzer::ConstructComplex(
    MaybeExpr &&real, MaybeExpr &&imaginary) {
  if (auto parts{common::AllPresent(std::move(real), std::move(imaginary))}) {
    if (auto converted{ConvertRealOperands(context.messages,
            std::move(std::get<0>(*parts)), std::move(std::get<1>(*parts)))}) {
      return {std::visit(
          [](auto &&pair) -> std::optional<Expr<SomeComplex>> {
            using realType = ResultType<decltype(pair[0])>;
            using zType = SameKind<TypeCategory::Complex, realType>;
            auto cmplx{ComplexConstructor<zType::kind>{
                std::move(pair[0]), std::move(pair[1])}};
            return {AsCategoryExpr(AsExpr(std::move(cmplx)))};
          },
          std::move(*converted))};
    }
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ComplexLiteralConstant &z) {
  return AsMaybeExpr(
      ConstructComplex(Analyze(std::get<0>(z.t)), Analyze(std::get<1>(z.t))));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::BOZLiteralConstant &x) {
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
  auto value{BOZLiteralConstant::ReadUnsigned(++p, base)};
  if (*p != '"') {
    context.messages.Say(
        "invalid digit ('%c') in BOZ literal %s"_err_en_US, *p, x.v.data());
    return std::nullopt;
  }
  if (value.overflow) {
    context.messages.Say("BOZ literal %s too large"_err_en_US, x.v.data());
    return std::nullopt;
  }
  return {AsGenericExpr(value.value)};
}

MaybeExpr ExprAnalyzer::Analyze(const parser::CharLiteralConstant &x) {
  int kind{Analyze(std::get<std::optional<parser::KindParam>>(x.t), 1)};
  auto value{std::get<std::string>(x.t)};
  auto result{common::SearchDynamicTypes(
      ConstantTypeVisitor<TypeCategory::Character, std::string>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    context.messages.Say("unsupported CHARACTER(KIND=%u)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::LogicalLiteralConstant &x) {
  auto kind{Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      defaults.defaultLogicalKind)};
  bool value{std::get<bool>(x.t)};
  auto result{common::SearchDynamicTypes(
      ConstantTypeVisitor<TypeCategory::Logical, bool>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    context.messages.Say("unsupported LOGICAL(KIND=%u)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Name &n) {
  if (n.symbol != nullptr) {
    auto *details{n.symbol->detailsIf<semantics::ObjectEntityDetails>()};
    if (details == nullptr ||
        !n.symbol->attrs().test(semantics::Attr::PARAMETER)) {
      context.messages.Say(
          "name (%s) is not a defined constant"_err_en_US, n.ToString().data());
      return std::nullopt;
    }
    // TODO: enumerators, do they have the PARAMETER attribute?
  }
  return std::nullopt;  // TODO parameters and enumerators
}

MaybeExpr ExprAnalyzer::Analyze(const parser::NamedConstant &n) {
  return Analyze(n.v);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Parentheses &x) {
  if (MaybeExpr operand{AnalyzeHelper(*this, *x.v)}) {
    return std::visit(
        common::visitors{
            [&](BOZLiteralConstant &&boz) {
              return operand;  // ignore parentheses around typeless
            },
            [](auto &&catExpr) {
              return std::visit(
                  [](auto &&expr) -> MaybeExpr {
                    using Ty = ResultType<decltype(expr)>;
                    if constexpr (common::HasMember<Parentheses<Ty>,
                                      decltype(expr.u)>) {
                      return {AsGenericExpr(
                          AsExpr(Parentheses<Ty>{std::move(expr)}))};
                    }
                    // TODO: support Parentheses in all Expr specializations
                    return std::nullopt;
                  },
                  std::move(catExpr.u));
            }},
        std::move(operand->u));
  }
  return std::nullopt;
}

// TODO: defined operators for illegal intrinsic operator cases
template<template<typename> class OPR, typename PARSED>
MaybeExpr BinaryOperationHelper(ExprAnalyzer &ea, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeHelper(ea, *std::get<0>(x.t)),
          AnalyzeHelper(ea, *std::get<1>(x.t)))}) {
    return NumericOperation<OPR>(ea.context.messages,
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both)));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Add &x) {
  return BinaryOperationHelper<Add>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Subtract &x) {
  return BinaryOperationHelper<Subtract>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Multiply &x) {
  return BinaryOperationHelper<Multiply>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Divide &x) {
  return BinaryOperationHelper<Divide>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::ComplexConstructor &x) {
  return AsMaybeExpr(ConstructComplex(AnalyzeHelper(*this, *std::get<0>(x.t)),
      AnalyzeHelper(*this, *std::get<1>(x.t))));
}

}  // namespace Fortran::evaluate

namespace Fortran::semantics {

MaybeExpr AnalyzeExpr(evaluate::FoldingContext &context,
    const IntrinsicTypeDefaultKinds &defaults, const parser::Expr &expr) {
  return evaluate::ExprAnalyzer{context, defaults}.Analyze(expr);
}

}  // namespace Fortran::semantics
