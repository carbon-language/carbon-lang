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
#include "dump-parse-tree.h"  // TODO pmk temporary
#include "symbol.h"
#include "../common/idioms.h"
#include "../evaluate/common.h"
#include "../evaluate/tools.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
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
  MaybeExpr Analyze(const parser::CharLiteralConstantSubstring &);
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
  MaybeExpr Analyze(const parser::Designator &);
  MaybeExpr Analyze(const parser::ArrayConstructor &);
  MaybeExpr Analyze(const parser::StructureConstructor &);
  MaybeExpr Analyze(const parser::TypeParamInquiry &);
  MaybeExpr Analyze(const parser::FunctionReference &);
  MaybeExpr Analyze(const parser::Expr::Parentheses &);
  MaybeExpr Analyze(const parser::Expr::UnaryPlus &);
  MaybeExpr Analyze(const parser::Expr::Negate &);
  MaybeExpr Analyze(const parser::Expr::NOT &);
  MaybeExpr Analyze(const parser::Expr::PercentLoc &);
  MaybeExpr Analyze(const parser::Expr::DefinedUnary &);
  MaybeExpr Analyze(const parser::Expr::Power &);
  MaybeExpr Analyze(const parser::Expr::Multiply &);
  MaybeExpr Analyze(const parser::Expr::Divide &);
  MaybeExpr Analyze(const parser::Expr::Add &);
  MaybeExpr Analyze(const parser::Expr::Subtract &);
  MaybeExpr Analyze(const parser::Expr::Concat &);
  MaybeExpr Analyze(const parser::Expr::LT &);
  MaybeExpr Analyze(const parser::Expr::LE &);
  MaybeExpr Analyze(const parser::Expr::EQ &);
  MaybeExpr Analyze(const parser::Expr::NE &);
  MaybeExpr Analyze(const parser::Expr::GE &);
  MaybeExpr Analyze(const parser::Expr::GT &);
  MaybeExpr Analyze(const parser::Expr::AND &);
  MaybeExpr Analyze(const parser::Expr::OR &);
  MaybeExpr Analyze(const parser::Expr::EQV &);
  MaybeExpr Analyze(const parser::Expr::NEQV &);
  MaybeExpr Analyze(const parser::Expr::XOR &);
  MaybeExpr Analyze(const parser::Expr::ComplexConstructor &);
  MaybeExpr Analyze(const parser::Expr::DefinedBinary &);

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

template<typename A>
MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const common::Indirection<A> &x) {
  return AnalyzeHelper(ea, *x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr &expr) {
  return std::visit(
      [&](const auto &x) { return AnalyzeHelper(*this, x); }, expr.u);
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
      case 'e': letterKind = defaults.defaultRealKind; break;
      case 'd': letterKind = defaults.defaultDoublePrecisionKind; break;
      case 'q': letterKind = defaults.defaultQuadPrecisionKind; break;
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

MaybeExpr ExprAnalyzer::Analyze(const parser::ComplexLiteralConstant &z) {
  return AsMaybeExpr(ConstructComplex(
      context.messages, Analyze(std::get<0>(z.t)), Analyze(std::get<1>(z.t))));
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

MaybeExpr ExprAnalyzer::Analyze(const parser::CharLiteralConstantSubstring &) {
  context.messages.Say(
      "pmk: CharLiteralConstantSubstring unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Designator &) {
  context.messages.Say("pmk: Designator unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ArrayConstructor &) {
  context.messages.Say("pmk: ArrayConstructor unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::StructureConstructor &) {
  context.messages.Say("pmk: StructureConstructor unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::TypeParamInquiry &) {
  context.messages.Say("pmk: TypeParamInquiry unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::FunctionReference &) {
  context.messages.Say("pmk: FunctionReference unimplemented\n"_err_en_US);
  return std::nullopt;
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

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::UnaryPlus &x) {
  return AnalyzeHelper(*this, *x.v);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Negate &x) {
  if (MaybeExpr operand{AnalyzeHelper(*this, *x.v)}) {
    return Negation(context.messages, std::move(operand->u));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NOT &) {
  context.messages.Say("pmk: NOT unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::PercentLoc &) {
  context.messages.Say("pmk: %LOC unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::DefinedUnary &) {
  context.messages.Say("pmk: DefinedUnary unimplemented\n"_err_en_US);
  return std::nullopt;
}

// TODO: check defined operators for illegal intrinsic operator cases
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
  return AsMaybeExpr(ConstructComplex(context.messages,
      AnalyzeHelper(*this, *std::get<0>(x.t)),
      AnalyzeHelper(*this, *std::get<1>(x.t))));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Power &) {
  context.messages.Say("pmk: Power unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Concat &) {
  context.messages.Say("pmk: Concat unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::LT &) {
  context.messages.Say("pmk: .LT. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::LE &) {
  context.messages.Say("pmk: .LE. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::EQ &) {
  context.messages.Say("pmk: .EQ. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NE &) {
  context.messages.Say("pmk: .NE. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::GT &) {
  context.messages.Say("pmk: .GT. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::GE &) {
  context.messages.Say("pmk: .GE. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::AND &) {
  context.messages.Say("pmk: .AND. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::OR &) {
  context.messages.Say("pmk: .OR. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::EQV &) {
  context.messages.Say("pmk: .EQV. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NEQV &) {
  context.messages.Say("pmk: .NEQV. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::XOR &) {
  context.messages.Say("pmk: .XOR. unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::DefinedBinary &) {
  context.messages.Say("pmk: DefinedBinary unimplemented\n"_err_en_US);
  return std::nullopt;
}

}  // namespace Fortran::evaluate

namespace Fortran::semantics {

MaybeExpr AnalyzeExpr(evaluate::FoldingContext &context,
    const IntrinsicTypeDefaultKinds &defaults, const parser::Expr &expr) {
  return evaluate::ExprAnalyzer{context, defaults}.Analyze(expr);
}

class Mutator {
public:
  Mutator(evaluate::FoldingContext &context,
      const IntrinsicTypeDefaultKinds &defaults, std::ostream &o)
    : context_{context}, defaults_{defaults}, out_{o} {}

  template<typename A> bool Pre(A &) { return true /* visit children */; }
  template<typename A> void Post(A &) {}

  bool Pre(parser::Expr &expr) {
    if (expr.typedExpr.get() == nullptr) {
      if (MaybeExpr checked{AnalyzeExpr(context_, defaults_, expr)}) {
        checked->Dump(out_ << "pmk checked: ") << '\n';
        expr.typedExpr.reset(
            new evaluate::GenericExprWrapper{std::move(*checked)});
      } else {
        out_ << "pmk: expression analysis failed for an expression: ";
        DumpTree(out_, expr);
      }
    }
    return false;
  }

private:
  evaluate::FoldingContext &context_;
  const IntrinsicTypeDefaultKinds &defaults_;
  std::ostream &out_;
};

void AnalyzeExpressions(parser::Program &program,
    evaluate::FoldingContext &context,
    const IntrinsicTypeDefaultKinds &defaults, std::ostream &o) {
  Mutator mutator{context, defaults, o};
  parser::Walk(program, mutator);
}

}  // namespace Fortran::semantics
