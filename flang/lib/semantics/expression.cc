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
#include "dump-parse-tree.h"  // TODO temporary
#include "semantics.h"
#include "symbol.h"
#include "../common/idioms.h"
#include "../evaluate/common.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/characters.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <functional>
#include <iostream>  // TODO pmk remove soon
#include <optional>

using namespace Fortran::parser::literals;

// Typedef for optional generic expressions (ubiquitous in this file)
using MaybeExpr =
    std::optional<Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>;

// Much of the code that implements semantic analysis of expressions is
// tightly coupled with their typed representations in lib/evaluate,
// and appears here in namespace Fortran::evaluate for convenience.
namespace Fortran::evaluate {

using common::TypeCategory;

// Constraint checking
void ExpressionAnalysisContext::CheckConstraints(MaybeExpr &expr) {
  if (inner_ != nullptr) {
    inner_->CheckConstraints(expr);
  }
  if (constraint_ != nullptr && expr.has_value()) {
    if (!(this->*constraint_)(*expr)) {
      expr.reset();
    }
  }
}

bool ExpressionAnalysisContext::ScalarConstraint(Expr<SomeType> &expr) {
  int rank{expr.Rank()};
  if (rank == 0) {
    return true;
  }
  Say("expression must be scalar, but has rank %d"_err_en_US, rank);
  return false;
}

bool ExpressionAnalysisContext::ConstantConstraint(Expr<SomeType> &expr) {
  expr = Fold(context_.foldingContext(), std::move(expr));
  if (IsConstant(expr)) {
    return true;
  }
  Say("expression must be constant"_err_en_US);
  return false;
}

bool ExpressionAnalysisContext::IntegerConstraint(Expr<SomeType> &expr) {
  if (std::holds_alternative<Expr<SomeInteger>>(expr.u)) {
    return true;
  }
  Say("expression must be INTEGER"_err_en_US);
  return false;
}

// A utility subroutine to repackage optional expressions of various levels
// of type specificity as fully general MaybeExpr values.
template<typename A> MaybeExpr AsMaybeExpr(A &&x) {
  return std::make_optional(AsGenericExpr(std::move(x)));
}
template<typename A> MaybeExpr AsMaybeExpr(std::optional<A> &&x) {
  if (x.has_value()) {
    return AsMaybeExpr(std::move(*x));
  }
  return std::nullopt;
}

// If a generic expression simply wraps a DataRef, extract it.
// TODO: put in tools.h?
template<typename A> std::optional<DataRef> ExtractDataRef(A &&) {
  return std::nullopt;
}

template<typename A> std::optional<DataRef> ExtractDataRef(Designator<A> &&d) {
  return std::visit(
      [](auto &&x) -> std::optional<DataRef> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (common::HasMember<Ty, decltype(DataRef::u)>) {
          return {DataRef{std::move(x)}};
        }
        return std::nullopt;
      },
      std::move(d.u));
}

template<TypeCategory CAT, int KIND>
std::optional<DataRef> ExtractDataRef(Expr<Type<CAT, KIND>> &&expr) {
  using Ty = ResultType<decltype(expr)>;
  if (auto *designator{std::get_if<Designator<Ty>>(&expr.u)}) {
    return ExtractDataRef(std::move(*designator));
  } else {
    return std::nullopt;
  }
}

template<TypeCategory CAT>
std::optional<DataRef> ExtractDataRef(Expr<SomeKind<CAT>> &&expr) {
  return std::visit(
      [](auto &&specificExpr) {
        return ExtractDataRef(std::move(specificExpr));
      },
      std::move(expr.u));
}

template<> std::optional<DataRef> ExtractDataRef(Expr<SomeType> &&expr) {
  return std::visit(
      common::visitors{
          [](BOZLiteralConstant &&) -> std::optional<DataRef> {
            return std::nullopt;
          },
          [](auto &&catExpr) { return ExtractDataRef(std::move(catExpr)); },
      },
      std::move(expr.u));
}

template<typename A>
std::optional<DataRef> ExtractDataRef(std::optional<A> &&x) {
  if (x.has_value()) {
    return ExtractDataRef(std::move(*x));
  }
  return std::nullopt;
}

struct CallAndArguments {
  ProcedureDesignator procedureDesignator;
  ActualArguments arguments;
};

// This local class wraps some state and a highly overloaded Analyze()
// member function that converts parse trees into (usually) generic
// expressions.
struct ExprAnalyzer : public ExpressionAnalysisContext {

  ExprAnalyzer(ExpressionAnalysisContext &eac)
    : ExpressionAnalysisContext{eac} {}

  MaybeExpr Analyze(const parser::Expr &);
  MaybeExpr Analyze(const parser::CharLiteralConstantSubstring &);
  MaybeExpr Analyze(const parser::LiteralConstant &);
  MaybeExpr Analyze(const parser::IntLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedIntLiteralConstant &);
  MaybeExpr Analyze(const parser::RealLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedRealLiteralConstant &);
  MaybeExpr Analyze(const parser::ComplexPart &);
  MaybeExpr Analyze(const parser::ComplexLiteralConstant &);
  MaybeExpr Analyze(const parser::CharLiteralConstant &);
  MaybeExpr Analyze(const parser::LogicalLiteralConstant &);
  MaybeExpr Analyze(const parser::HollerithLiteralConstant &);
  MaybeExpr Analyze(const parser::BOZLiteralConstant &);
  MaybeExpr Analyze(const parser::Name &);
  MaybeExpr Analyze(const parser::NamedConstant &);
  MaybeExpr Analyze(const parser::Substring &);
  MaybeExpr Analyze(const parser::ArrayElement &);
  MaybeExpr Analyze(const parser::StructureComponent &);
  MaybeExpr Analyze(const parser::CoindexedNamedObject &);
  MaybeExpr Analyze(const parser::ArrayConstructor &);
  MaybeExpr Analyze(const parser::StructureConstructor &);
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
  MaybeExpr Analyze(const parser::FunctionReference &);

  // Kind parameter analysis always returns a valid kind value.
  int Analyze(
      const std::optional<parser::KindParam> &, int defaultKind, int kanjiKind);

  MaybeExpr Analyze(std::string &&, int);

  std::optional<Subscript> Analyze(const parser::SectionSubscript &);
  std::vector<Subscript> Analyze(const std::list<parser::SectionSubscript> &);

  std::optional<Expr<SubscriptInteger>> AsSubscript(MaybeExpr &&);
  std::optional<Expr<SubscriptInteger>> GetSubstringBound(
      const std::optional<parser::ScalarIntExpr> &);
  std::optional<Expr<SubscriptInteger>> TripletPart(
      const std::optional<parser::Subscript> &);
  MaybeExpr ApplySubscripts(DataRef &&, std::vector<Subscript> &&);
  MaybeExpr CompleteSubscripts(ArrayRef &&);

  MaybeExpr TopLevelChecks(DataRef &&);
  void CheckUnsubscriptedComponent(const Component &);

  std::optional<CallAndArguments> Procedure(
      const parser::ProcedureDesignator &, ActualArguments &);
};

MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr &expr) {
  ExprAnalyzer ea{context};
  return ea.Analyze(expr);
}

template<typename... As>
MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const std::variant<As...> &u) {
  return std::visit([&](const auto &x) { return AnalyzeExpr(context, x); }, u);
}

template<typename A>
MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const common::Indirection<A> &x) {
  return AnalyzeExpr(context, *x);
}

template<>
MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Designator &d) {
  // These checks have to be deferred to these "top level" data-refs where
  // we can be sure that there are no following subscripts (yet).
  if (MaybeExpr result{AnalyzeExpr(context, d.u)}) {
    if (std::optional<evaluate::DataRef> dataRef{
            evaluate::ExtractDataRef(std::move(result))}) {
      evaluate::ExprAnalyzer ea{context};
      return ea.TopLevelChecks(std::move(*dataRef));
    }
    return result;
  }
  return std::nullopt;
}

template<typename A>
MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context, const A &x) {
  if constexpr (UnionTrait<A>) {
    return AnalyzeExpr(context, x.u);
  } else {
    evaluate::ExprAnalyzer ea{context};
    MaybeExpr result{ea.Analyze(x)};
    ea.CheckConstraints(result);
    return result;
  }
}

template MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr &);

// Implementations of ExprAnalyzer::Analyze follow for various parse tree
// node types.

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr &expr) {
  if (!expr.source.empty()) {
    // Analyze the expression in a specified source position context for better
    // error reporting.
    auto save{context_.foldingContext().messages.SetLocation(expr.source)};
    MaybeExpr result{AnalyzeExpr(*this, expr.u)};
    CheckConstraints(result);
    return result;
  } else {
    MaybeExpr result{AnalyzeExpr(*this, expr.u)};
    CheckConstraints(result);
    return result;
  }
}

int ExprAnalyzer::Analyze(const std::optional<parser::KindParam> &kindParam,
    int defaultKind, int kanjiKind = -1) {
  if (!kindParam.has_value()) {
    return defaultKind;
  }
  return std::visit(
      common::visitors{
          [](std::uint64_t k) { return static_cast<int>(k); },
          [&](const parser::Scalar<
              parser::Integer<parser::Constant<parser::Name>>> &n) {
            if (MaybeExpr ie{AnalyzeExpr(*this, n)}) {
              if (std::optional<std::int64_t> i64{ToInt64(*ie)}) {
                int iv = *i64;
                if (iv == *i64) {
                  return iv;
                }
              }
            }
            Say("KIND type parameter must be a scalar integer constant"_err_en_US);
            return defaultKind;
          },
          [&](parser::KindParam::Kanji) {
            if (kanjiKind >= 0) {
              return kanjiKind;
            }
            Say("Kanji not allowed here"_err_en_US);
            return defaultKind;
          },
      },
      kindParam->u);
}

// Common handling of parser::IntLiteralConstant and SignedIntLiteralConstant
template<typename PARSED>
MaybeExpr IntLiteralConstant(ExprAnalyzer &ea, const PARSED &x) {
  int kind{ea.Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      ea.context().defaultKinds().GetDefaultKind(TypeCategory::Integer))};
  auto value{std::get<0>(x.t)};  // std::(u)int64_t
  auto result{common::SearchDynamicTypes(
      TypeKindVisitor<TypeCategory::Integer, Constant, std::int64_t>{
          kind, static_cast<std::int64_t>(value)})};
  if (!result.has_value()) {
    ea.Say("unsupported INTEGER(KIND=%d)"_err_en_US, kind);
  }
  return result;
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

struct RealTypeVisitor {
  using Result = std::optional<Expr<SomeReal>>;
  static constexpr std::size_t Types{std::tuple_size_v<RealTypes>};

  RealTypeVisitor(int k, parser::CharBlock lit, FoldingContext &ctx)
    : kind{k}, literal{lit}, context{ctx} {}

  template<std::size_t J> Result Test() {
    using Ty = std::tuple_element_t<J, RealTypes>;
    if (kind == Ty::kind) {
      return {AsCategoryExpr(ReadRealLiteral<Ty>(literal, context))};
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
  auto save{context_.foldingContext().messages.SetLocation(x.real.source)};
  // If a kind parameter appears, it defines the kind of the literal and any
  // letter used in an exponent part (e.g., the 'E' in "6.02214E+23")
  // should agree.  In the absence of an explicit kind parameter, any exponent
  // letter determines the kind.  Otherwise, defaults apply.
  auto &defaults{context_.defaultKinds()};
  int defaultKind{defaults.GetDefaultKind(TypeCategory::Real)};
  const char *end{x.real.source.end()};
  std::optional<int> letterKind;
  for (const char *p{x.real.source.begin()}; p < end; ++p) {
    if (parser::IsLetter(*p)) {
      switch (*p) {
      case 'e': letterKind = defaults.GetDefaultKind(TypeCategory::Real); break;
      case 'd': letterKind = defaults.doublePrecisionKind(); break;
      case 'q': letterKind = defaults.quadPrecisionKind(); break;
      default: Say("unknown exponent letter '%c'"_err_en_US, *p);
      }
      break;
    }
  }
  if (letterKind.has_value()) {
    defaultKind = *letterKind;
  }
  auto kind{Analyze(x.kind, defaultKind)};
  if (letterKind.has_value() && kind != *letterKind) {
    Say("explicit kind parameter on real constant disagrees with exponent letter"_en_US);
  }
  auto result{common::SearchDynamicTypes(
      RealTypeVisitor{kind, x.real.source, context_.foldingContext()})};
  if (!result.has_value()) {
    Say("unsupported REAL(KIND=%d)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::SignedRealLiteralConstant &x) {
  if (MaybeExpr result{Analyze(std::get<parser::RealLiteralConstant>(x.t))}) {
    auto *realExpr{std::get_if<Expr<SomeReal>>(&result->u)};
    CHECK(realExpr != nullptr);
    if (auto sign{std::get<std::optional<parser::Sign>>(x.t)}) {
      if (sign == parser::Sign::Negative) {
        return {AsGenericExpr(-std::move(*realExpr))};
      }
    }
    return result;
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ComplexPart &x) {
  return AnalyzeExpr(*this, x.u);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ComplexLiteralConstant &z) {
  return AsMaybeExpr(ConstructComplex(context_.foldingContext().messages,
      Analyze(std::get<0>(z.t)), Analyze(std::get<1>(z.t)),
      context_.defaultKinds().GetDefaultKind(TypeCategory::Real)));
}

MaybeExpr ExprAnalyzer::Analyze(std::string &&string, int kind) {
  if (!IsValidKindOfIntrinsicType(TypeCategory::Character, kind)) {
    Say("unsupported CHARACTER(KIND=%d)"_err_en_US, kind);
    return std::nullopt;
  }
  if (kind == 1) {
    return {AsGenericExpr(
        Constant<Type<TypeCategory::Character, 1>>{std::move(string)})};
  } else if (std::optional<std::u32string> unicode{
                 parser::DecodeUTF8(string)}) {
    if (kind == 4) {
      return {AsGenericExpr(
          Constant<Type<TypeCategory::Character, 4>>{std::move(*unicode)})};
    }
    CHECK(kind == 2);
    // TODO: better Kanji support
    std::u16string result;
    for (const char32_t &ch : *unicode) {
      result += static_cast<char16_t>(ch);
    }
    return {AsGenericExpr(
        Constant<Type<TypeCategory::Character, 2>>{std::move(result)})};
  } else {
    Say("bad UTF-8 encoding of CHARACTER(KIND=%d) literal"_err_en_US, kind);
    return std::nullopt;
  }
}

MaybeExpr ExprAnalyzer::Analyze(const parser::CharLiteralConstant &x) {
  int kind{Analyze(std::get<std::optional<parser::KindParam>>(x.t), 1)};
  auto value{std::get<std::string>(x.t)};
  return Analyze(std::move(value), kind);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::LogicalLiteralConstant &x) {
  auto kind{Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      context_.defaultKinds().GetDefaultKind(TypeCategory::Logical))};
  bool value{std::get<bool>(x.t)};
  auto result{common::SearchDynamicTypes(
      TypeKindVisitor<TypeCategory::Logical, Constant, bool>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    Say("unsupported LOGICAL(KIND=%d)"_err_en_US, kind);
  }
  return result;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::HollerithLiteralConstant &x) {
  int kind{context_.defaultKinds().GetDefaultKind(TypeCategory::Character)};
  auto value{x.v};
  return Analyze(std::move(value), kind);
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
    Say("invalid digit ('%c') in BOZ literal %s"_err_en_US, *p, x.v.data());
    return std::nullopt;
  }
  if (value.overflow) {
    Say("BOZ literal %s too large"_err_en_US, x.v.data());
    return std::nullopt;
  }
  return {AsGenericExpr(std::move(value.value))};
}

// Wraps a object in an explicitly typed representation (e.g., Designator<>
// or FunctionRef<>) as instantiated on a dynamic type.
// TODO: move to tools.h?
template<TypeCategory CATEGORY, template<typename> typename WRAPPER,
    typename WRAPPED>
MaybeExpr WrapperHelper(int kind, WRAPPED &&x) {
  return common::SearchDynamicTypes(
      TypeKindVisitor<CATEGORY, WRAPPER, WRAPPED>{kind, std::move(x)});
}

template<template<typename> typename WRAPPER, typename WRAPPED>
MaybeExpr TypedWrapper(DynamicType &&dyType, WRAPPED &&x) {
  switch (dyType.category) {
  case TypeCategory::Integer:
    return WrapperHelper<TypeCategory::Integer, WRAPPER, WRAPPED>(
        dyType.kind, std::move(x));
  case TypeCategory::Real:
    return WrapperHelper<TypeCategory::Real, WRAPPER, WRAPPED>(
        dyType.kind, std::move(x));
  case TypeCategory::Complex:
    return WrapperHelper<TypeCategory::Complex, WRAPPER, WRAPPED>(
        dyType.kind, std::move(x));
  case TypeCategory::Character:
    return WrapperHelper<TypeCategory::Character, WRAPPER, WRAPPED>(
        dyType.kind, std::move(x));
  case TypeCategory::Logical:
    return WrapperHelper<TypeCategory::Logical, WRAPPER, WRAPPED>(
        dyType.kind, std::move(x));
  case TypeCategory::Derived:
    return AsGenericExpr(Expr<SomeDerived>{WRAPPER<SomeDerived>{std::move(x)}});
  default: CRASH_NO_CASE;
  }
}

// Wraps a data reference in a typed Designator<>.
static MaybeExpr Designate(DataRef &&dataRef) {
  const Symbol &symbol{dataRef.GetLastSymbol()};
  if (std::optional<DynamicType> dyType{GetSymbolType(symbol)}) {
    return TypedWrapper<Designator, DataRef>(
        std::move(*dyType), std::move(dataRef));
  }
  // TODO: graceful errors on CLASS(*) and TYPE(*) misusage
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Name &n) {
  if (n.symbol == nullptr) {
    Say(n.source,
        "TODO INTERNAL: name '%s' was not resolved to a symbol"_err_en_US,
        n.ToString().data());
  } else if (n.symbol->attrs().test(semantics::Attr::PARAMETER)) {
    if (auto *details{n.symbol->detailsIf<semantics::ObjectEntityDetails>()}) {
      auto &init{details->init()};
      if (init.Resolve(context_)) {
        return init.Get();
      }
    }
    Say(n.source, "parameter '%s' does not have a value"_err_en_US,
        n.ToString().data());
    // TODO: enumerators, do they have the PARAMETER attribute?
  } else {
    if (MaybeExpr result{Designate(DataRef{*n.symbol})}) {
      return result;
    }
    Say(n.source, "not of a supported type and kind"_err_en_US);
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::NamedConstant &n) {
  if (MaybeExpr value{Analyze(n.v)}) {
    Expr<SomeType> folded{Fold(context_.foldingContext(), std::move(*value))};
    if (IsConstant(folded)) {
      return {folded};
    }
    Say(n.v.source, "must be a constant"_err_en_US);
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Substring &ss) {
  if (MaybeExpr baseExpr{AnalyzeExpr(*this, std::get<parser::DataRef>(ss.t))}) {
    if (std::optional<DataRef> dataRef{ExtractDataRef(std::move(*baseExpr))}) {
      if (MaybeExpr newBaseExpr{TopLevelChecks(std::move(*dataRef))}) {
        if (std::optional<DataRef> checked{
                ExtractDataRef(std::move(*newBaseExpr))}) {
          const parser::SubstringRange &range{
              std::get<parser::SubstringRange>(ss.t)};
          std::optional<Expr<SubscriptInteger>> first{
              GetSubstringBound(std::get<0>(range.t))};
          std::optional<Expr<SubscriptInteger>> last{
              GetSubstringBound(std::get<1>(range.t))};
          const Symbol &symbol{checked->GetLastSymbol()};
          if (std::optional<DynamicType> dynamicType{GetSymbolType(symbol)}) {
            if (dynamicType->category == TypeCategory::Character) {
              return WrapperHelper<TypeCategory::Character, Designator,
                  Substring>(dynamicType->kind,
                  Substring{
                      std::move(*checked), std::move(first), std::move(last)});
            }
          }
          Say("substring may apply only to CHARACTER"_err_en_US);
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> ExprAnalyzer::AsSubscript(
    MaybeExpr &&expr) {
  if (expr.has_value()) {
    if (expr->Rank() > 1) {
      Say("subscript expression has rank %d"_err_en_US, expr->Rank());
    }
    if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
      if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
        return {std::move(*ssIntExpr)};
      }
      return {Expr<SubscriptInteger>{
          Convert<SubscriptInteger, TypeCategory::Integer>{
              std::move(*intExpr)}}};
    } else {
      Say("subscript expression is not INTEGER"_err_en_US);
    }
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> ExprAnalyzer::GetSubstringBound(
    const std::optional<parser::ScalarIntExpr> &bound) {
  if (bound.has_value()) {
    if (MaybeExpr expr{AnalyzeExpr(*this, *bound)}) {
      if (expr->Rank() > 1) {
        Say("substring bound expression has rank %d"_err_en_US, expr->Rank());
      }
      if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
        if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
          return {std::move(*ssIntExpr)};
        }
        return {Expr<SubscriptInteger>{
            Convert<SubscriptInteger, TypeCategory::Integer>{
                std::move(*intExpr)}}};
      } else {
        Say("substring bound expression is not INTEGER"_err_en_US);
      }
    }
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> ExprAnalyzer::TripletPart(
    const std::optional<parser::Subscript> &s) {
  if (s.has_value()) {
    return AsSubscript(AnalyzeExpr(*this, *s));
  }
  return std::nullopt;
}

std::optional<Subscript> ExprAnalyzer::Analyze(
    const parser::SectionSubscript &ss) {
  return std::visit(
      common::visitors{
          [&](const parser::SubscriptTriplet &t) {
            return std::make_optional(Subscript{Triplet{
                TripletPart(std::get<0>(t.t)), TripletPart(std::get<1>(t.t)),
                TripletPart(std::get<2>(t.t))}});
          },
          [&](const auto &s) -> std::optional<Subscript> {
            if (auto subscriptExpr{AsSubscript(AnalyzeExpr(*this, s))}) {
              return {Subscript{std::move(*subscriptExpr)}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

std::vector<Subscript> ExprAnalyzer::Analyze(
    const std::list<parser::SectionSubscript> &sss) {
  std::vector<Subscript> subscripts;
  for (const auto &s : sss) {
    if (auto subscript{Analyze(s)}) {
      subscripts.emplace_back(std::move(*subscript));
    }
  }
  return subscripts;
}

MaybeExpr ExprAnalyzer::ApplySubscripts(
    DataRef &&dataRef, std::vector<Subscript> &&subscripts) {
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) {
            return CompleteSubscripts(ArrayRef{*symbol, std::move(subscripts)});
          },
          [&](auto &&base) -> MaybeExpr {
            using Ty = std::decay_t<decltype(base)>;
            if constexpr (common::HasMember<Ty, decltype(ArrayRef::u)>) {
              return CompleteSubscripts(
                  ArrayRef{std::move(base), std::move(subscripts)});
            }
            return std::nullopt;
          },
      },
      std::move(dataRef.u));
}

MaybeExpr ExprAnalyzer::CompleteSubscripts(ArrayRef &&ref) {
  const Symbol &symbol{ref.GetLastSymbol()};
  int symbolRank{symbol.Rank()};
  if (ref.subscript.empty()) {
    // A -> A(:,:)
    for (int j{0}; j < symbolRank; ++j) {
      ref.subscript.emplace_back(Subscript{Triplet{}});
    }
  }
  int subscripts = ref.subscript.size();
  if (subscripts != symbolRank) {
    Say("reference to rank-%d object '%s' has %d subscripts"_err_en_US,
        symbolRank, symbol.name().ToString().data(), subscripts);
  } else if (Component * component{std::get_if<Component>(&ref.u)}) {
    int baseRank{component->Rank()};
    if (baseRank > 0) {
      int rank{ref.Rank()};
      if (rank > 0) {
        Say("subscripts of rank-%d component reference have rank %d, but must all be scalar"_err_en_US,
            baseRank, rank);
      }
    }
  } else if (const auto *details{
                 symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    // C928 & C1002
    if (Triplet * last{std::get_if<Triplet>(&ref.subscript.back().u)}) {
      if (!last->upper().has_value() && details->IsAssumedSize()) {
        Say("assumed-size array '%s' must have explicit final subscript upper bound value"_err_en_US,
            symbol.name().ToString().data());
      }
    }
  }
  return Designate(DataRef{std::move(ref)});
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ArrayElement &ae) {
  std::vector<Subscript> subscripts{Analyze(ae.subscripts)};
  if (MaybeExpr baseExpr{AnalyzeExpr(*this, ae.base)}) {
    if (std::optional<DataRef> dataRef{ExtractDataRef(std::move(*baseExpr))}) {
      if (MaybeExpr result{
              ApplySubscripts(std::move(*dataRef), std::move(subscripts))}) {
        return result;
      }
    }
  }
  Say("subscripts may be applied only to an object or component"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::StructureComponent &sc) {
  if (MaybeExpr base{AnalyzeExpr(*this, sc.base)}) {
    if (auto *dtExpr{std::get_if<Expr<SomeDerived>>(&base->u)}) {
      Symbol *sym{sc.component.symbol};
      const semantics::DerivedTypeSpec *dtSpec{nullptr};
      if (std::optional<DynamicType> dtDyTy{dtExpr->GetType()}) {
        dtSpec = dtDyTy->derived;
      }
      if (sym == nullptr) {
        Say(sc.component.source,
            "component name was not resolved to a symbol"_err_en_US);
      } else if (sym->detailsIf<semantics::TypeParamDetails>()) {
        Say(sc.component.source,
            "TODO: type parameter inquiry unimplemented"_err_en_US);
      } else if (dtSpec == nullptr) {
        Say(sc.component.source,
            "TODO: base of component reference lacks a derived type"_err_en_US);
      } else if (&sym->owner() != dtSpec->scope()) {
        // TODO: extended derived types - insert explicit reference to base?
        Say(sc.component.source,
            "component is not in scope of derived TYPE(%s)"_err_en_US,
            dtSpec->name().ToString().data());
      } else if (std::optional<DataRef> dataRef{
                     ExtractDataRef(std::move(*dtExpr))}) {
        Component component{std::move(*dataRef), *sym};
        return Designate(DataRef{std::move(component)});
      } else {
        Say(sc.component.source,
            "base of component reference must be a data reference"_err_en_US);
      }
    } else if (auto *zExpr{std::get_if<Expr<SomeComplex>>(&base->u)}) {
      ComplexPart::Part part{ComplexPart::Part::RE};
      if (sc.component.source == parser::CharBlock{"im", 2}) {
        part = ComplexPart::Part::IM;
      } else if (sc.component.source != parser::CharBlock{"re", 2}) {
        Say(sc.component.source,
            "component of complex value must be %%RE or %%IM"_err_en_US);
        return std::nullopt;
      }
      if (std::optional<DataRef> dataRef{ExtractDataRef(std::move(*zExpr))}) {
        Expr<SomeReal> realExpr{std::visit(
            [&](const auto &z) {
              using PartType = typename ResultType<decltype(z)>::Part;
              return AsCategoryExpr(
                  Designator<PartType>{ComplexPart{std::move(*dataRef), part}});
            },
            zExpr->u)};
        return {AsGenericExpr(std::move(realExpr))};
      }
    } else {
      Say("derived type required before '%%%s'"_err_en_US,
          sc.component.ToString().data());
    }
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::CoindexedNamedObject &co) {
  // TODO: CheckUnsubscriptedComponent or its equivalent
  Say("TODO: CoindexedNamedObject unimplemented"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::CharLiteralConstantSubstring &x) {
  const parser::SubstringRange &range{std::get<parser::SubstringRange>(x.t)};
  std::optional<Expr<SubscriptInteger>> lower{
      GetSubstringBound(std::get<0>(range.t))};
  std::optional<Expr<SubscriptInteger>> upper{
      GetSubstringBound(std::get<1>(range.t))};
  if (MaybeExpr string{Analyze(std::get<parser::CharLiteralConstant>(x.t))}) {
    if (auto *charExpr{std::get_if<Expr<SomeCharacter>>(&string->u)}) {
      Expr<SubscriptInteger> length{std::visit(
          [](const auto &ckExpr) { return ckExpr.LEN(); }, charExpr->u)};
      if (!lower.has_value()) {
        lower = Expr<SubscriptInteger>{1};
      }
      if (!upper.has_value()) {
        std::optional<std::int64_t> size{ToInt64(length)};
        CHECK(size.has_value());
        upper = Expr<SubscriptInteger>{static_cast<std::int64_t>(*size)};
      }
      return std::visit(
          [&](auto &&ckExpr) -> MaybeExpr {
            using Result = ResultType<decltype(ckExpr)>;
            auto *cp{std::get_if<Constant<Result>>(&ckExpr.u)};
            CHECK(cp != nullptr);  // the parent was parsed as a constant string
            StaticDataObject::Pointer staticData{StaticDataObject::Create()};
            staticData->set_alignment(Result::kind)
                .set_itemBytes(Result::kind)
                .Push(cp->value);
            Substring substring{
                std::move(staticData), std::move(*lower), std::move(*upper)};
            return AsGenericExpr(Expr<SomeCharacter>{
                Expr<Result>{Designator<Result>{std::move(substring)}}});
          },
          std::move(charExpr->u));
    }
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ArrayConstructor &) {
  Say("TODO: ArrayConstructor unimplemented"_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::StructureConstructor &) {
  Say("TODO: StructureConstructor unimplemented"_err_en_US);
  return std::nullopt;
}

std::optional<CallAndArguments> ExprAnalyzer::Procedure(
    const parser::ProcedureDesignator &pd, ActualArguments &arguments) {
  return std::visit(
      common::visitors{
          [&](const parser::Name &n) -> std::optional<CallAndArguments> {
            if (n.symbol == nullptr) {
              Say("TODO INTERNAL no symbol for procedure designator name '%s'"_err_en_US,
                  n.ToString().data());
              return std::nullopt;
            }
            return std::visit(
                common::visitors{
                    [&](const semantics::ProcEntityDetails &p)
                        -> std::optional<CallAndArguments> {
                      if (p.HasExplicitInterface()) {
                        // TODO: check actual arguments vs. interface
                      } else {
                        CallCharacteristics cc{n.source};
                        if (std::optional<SpecificCall> specificCall{
                                context_.intrinsics().Probe(cc, arguments,
                                    &context_.foldingContext().messages)}) {
                          return {CallAndArguments{
                              ProcedureDesignator{
                                  std::move(specificCall->specificIntrinsic)},
                              std::move(specificCall->arguments)}};
                        } else {
                          // TODO: if name is not INTRINSIC, call with implicit
                          // interface
                        }
                      }
                      return {CallAndArguments{ProcedureDesignator{*n.symbol},
                          std::move(arguments)}};
                    },
                    [&](const auto &) -> std::optional<CallAndArguments> {
                      Say("TODO: unimplemented/invalid kind of symbol as procedure designator '%s'"_err_en_US,
                          n.ToString().data());
                      return std::nullopt;
                    },
                },
                n.symbol->details());
          },
          [&](const parser::ProcComponentRef &pcr)
              -> std::optional<CallAndArguments> {
            if (MaybeExpr component{AnalyzeExpr(*this, pcr.v)}) {
              // TODO distinguish PCR from TBP
              // TODO optional PASS argument for TBP
              Say("TODO: proc component ref"_err_en_US);
              return std::nullopt;
            } else {
              return std::nullopt;
            }
          },
      },
      pd.u);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::FunctionReference &funcRef) {
  // TODO: C1002: Allow a whole assumed-size array to appear if the dummy
  // argument would accept it.  Handle by special-casing the context
  // ActualArg -> Variable -> Designator.
  ActualArguments arguments;
  for (const auto &arg :
      std::get<std::list<parser::ActualArgSpec>>(funcRef.v.t)) {
    MaybeExpr actualArgExpr;
    std::visit(
        common::visitors{
            [&](const common::Indirection<parser::Variable> &v) {
              actualArgExpr = AnalyzeExpr(*this, v);
            },
            [&](const common::Indirection<parser::Expr> &x) {
              actualArgExpr = Analyze(*x);
            },
            [&](const parser::Name &n) {
              Say("TODO: procedure name actual arg"_err_en_US);
            },
            [&](const parser::ProcComponentRef &) {
              Say("TODO: proc component ref actual arg"_err_en_US);
            },
            [&](const parser::AltReturnSpec &) {
              Say("alternate return specification cannot appear on function reference"_err_en_US);
            },
            [&](const parser::ActualArg::PercentRef &) {
              Say("TODO: %REF() argument"_err_en_US);
            },
            [&](const parser::ActualArg::PercentVal &) {
              Say("TODO: %VAL() argument"_err_en_US);
            },
        },
        std::get<parser::ActualArg>(arg.t).u);
    if (actualArgExpr.has_value()) {
      arguments.emplace_back(std::make_optional(
          Fold(context_.foldingContext(), std::move(*actualArgExpr))));
      if (const auto &argKW{std::get<std::optional<parser::Keyword>>(arg.t)}) {
        arguments.back()->keyword = argKW->v.source;
      }
    } else {
      return std::nullopt;
    }
  }

  // TODO: map user generic to specific procedure
  // TODO: validate arguments against user interface
  if (std::optional<CallAndArguments> proc{Procedure(
          std::get<parser::ProcedureDesignator>(funcRef.v.t), arguments)}) {
    if (std::optional<DynamicType> dyType{
            proc->procedureDesignator.GetType()}) {
      return TypedWrapper<FunctionRef, ProcedureRef>(std::move(*dyType),
          ProcedureRef{std::move(proc->procedureDesignator),
              std::move(proc->arguments)});
    }
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Parentheses &x) {
  // TODO: C1003: A parenthesized function reference may not return a
  // procedure pointer.
  if (MaybeExpr operand{AnalyzeExpr(*this, *x.v)}) {
    return std::visit(
        common::visitors{
            [&](BOZLiteralConstant &&boz) {
              return operand;  // ignore parentheses around typeless constants
            },
            [&](Expr<SomeDerived> &&) {
              // TODO: parenthesized derived type variable
              return operand;
            },
            [](auto &&catExpr) {
              return std::visit(
                  [](auto &&expr) -> MaybeExpr {
                    using Ty = ResultType<decltype(expr)>;
                    return {AsGenericExpr(Parentheses<Ty>{std::move(expr)})};
                  },
                  std::move(catExpr.u));
            },
        },
        std::move(operand->u));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::UnaryPlus &x) {
  MaybeExpr value{AnalyzeExpr(*this, *x.v)};
  if (value.has_value()) {
    std::visit(
        common::visitors{
            [](const BOZLiteralConstant &) {},  // allow +Z'1', it's harmless
            [&](const auto &catExpr) {
              TypeCategory cat{ResultType<decltype(catExpr)>::category};
              if (cat != TypeCategory::Integer && cat != TypeCategory::Real &&
                  cat != TypeCategory::Complex) {
                Say("operand of unary + must be of a numeric type"_err_en_US);
              }
            },
        },
        value->u);
  }
  return value;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Negate &x) {
  if (MaybeExpr operand{AnalyzeExpr(*this, *x.v)}) {
    return Negation(context_.foldingContext().messages, std::move(*operand));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NOT &x) {
  if (MaybeExpr operand{AnalyzeExpr(*this, *x.v)}) {
    return std::visit(
        common::visitors{
            [](Expr<SomeLogical> &&lx) -> MaybeExpr {
              return {AsGenericExpr(LogicalNegation(std::move(lx)))};
            },
            [=](auto &&) -> MaybeExpr {
              // TODO: accept INTEGER operand and maybe typeless
              // if not overridden
              Say("Operand of .NOT. must be LOGICAL"_err_en_US);
              return std::nullopt;
            },
        },
        std::move(operand->u));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::PercentLoc &) {
  Say("TODO: %LOC unimplemented"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::DefinedUnary &) {
  Say("TODO: DefinedUnary unimplemented"_err_en_US);
  return std::nullopt;
}

// TODO: check defined operators for illegal intrinsic operator cases
template<template<typename> class OPR, typename PARSED>
MaybeExpr BinaryOperationHelper(ExprAnalyzer &ea, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(ea, *std::get<0>(x.t)),
          AnalyzeExpr(ea, *std::get<1>(x.t)))}) {
    int leftRank{std::get<0>(*both).Rank()};
    int rightRank{std::get<1>(*both).Rank()};
    if (leftRank > 0 && rightRank > 0 && leftRank != rightRank) {
      ea.Say("left operand has rank %d, right operand has rank %d"_err_en_US,
          leftRank, rightRank);
    }
    return NumericOperation<OPR>(ea.context().foldingContext().messages,
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both)),
        ea.context().defaultKinds().GetDefaultKind(TypeCategory::Real));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Power &x) {
  return BinaryOperationHelper<Power>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Multiply &x) {
  return BinaryOperationHelper<Multiply>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Divide &x) {
  return BinaryOperationHelper<Divide>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Add &x) {
  return BinaryOperationHelper<Add>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Subtract &x) {
  return BinaryOperationHelper<Subtract>(*this, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::ComplexConstructor &x) {
  return AsMaybeExpr(ConstructComplex(context_.foldingContext().messages,
      AnalyzeExpr(*this, *std::get<0>(x.t)),
      AnalyzeExpr(*this, *std::get<1>(x.t)),
      context_.defaultKinds().GetDefaultKind(TypeCategory::Real)));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Concat &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(*this, *std::get<0>(x.t)),
          AnalyzeExpr(*this, *std::get<1>(x.t)))}) {
    return std::visit(
        common::visitors{
            [&](Expr<SomeCharacter> &&cx, Expr<SomeCharacter> &&cy) {
              return std::visit(
                  [&](auto &&cxk, auto &&cyk) -> MaybeExpr {
                    using Ty = ResultType<decltype(cxk)>;
                    if constexpr (std::is_same_v<Ty,
                                      ResultType<decltype(cyk)>>) {
                      return {AsGenericExpr(
                          Concat<Ty::kind>{std::move(cxk), std::move(cyk)})};
                    } else {
                      Say("Operands of // must be the same kind of CHARACTER"_err_en_US);
                      return std::nullopt;
                    }
                  },
                  std::move(cx.u), std::move(cy.u));
            },
            [&](auto &&, auto &&) -> MaybeExpr {
              Say("Operands of // must be CHARACTER"_err_en_US);
              return std::nullopt;
            },
        },
        std::move(std::get<0>(*both).u), std::move(std::get<1>(*both).u));
  }
  return std::nullopt;
}

// TODO: check defined operators for illegal intrinsic operator cases
template<typename PARSED>
MaybeExpr RelationHelper(
    ExprAnalyzer &ea, RelationalOperator opr, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(ea, *std::get<0>(x.t)),
          AnalyzeExpr(ea, *std::get<1>(x.t)))}) {
    return AsMaybeExpr(Relate(ea.context().foldingContext().messages, opr,
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both))));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::LT &x) {
  return RelationHelper(*this, RelationalOperator::LT, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::LE &x) {
  return RelationHelper(*this, RelationalOperator::LE, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::EQ &x) {
  return RelationHelper(*this, RelationalOperator::EQ, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NE &x) {
  return RelationHelper(*this, RelationalOperator::NE, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::GE &x) {
  return RelationHelper(*this, RelationalOperator::GE, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::GT &x) {
  return RelationHelper(*this, RelationalOperator::GT, x);
}

// TODO: check defined operators for illegal intrinsic operator cases
template<typename PARSED>
MaybeExpr LogicalHelper(
    ExprAnalyzer &ea, LogicalOperator opr, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(ea, *std::get<0>(x.t)),
          AnalyzeExpr(ea, *std::get<1>(x.t)))}) {
    return std::visit(
        common::visitors{
            [=](Expr<SomeLogical> &&lx, Expr<SomeLogical> &&ly) -> MaybeExpr {
              return {AsGenericExpr(
                  BinaryLogicalOperation(opr, std::move(lx), std::move(ly)))};
            },
            [&](auto &&, auto &&) -> MaybeExpr {
              // TODO: extension: INTEGER and typeless operands
              // ifort and PGI accept them if not overridden
              // need to define IAND, IOR, IEOR intrinsic representation
              ea.Say("operands to LOGICAL operation must be LOGICAL"_err_en_US);
              return {};
            },
        },
        std::move(std::get<0>(*both).u), std::move(std::get<1>(*both).u));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::AND &x) {
  return LogicalHelper(*this, LogicalOperator::And, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::OR &x) {
  return LogicalHelper(*this, LogicalOperator::Or, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::EQV &x) {
  return LogicalHelper(*this, LogicalOperator::Eqv, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NEQV &x) {
  return LogicalHelper(*this, LogicalOperator::Neqv, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::XOR &x) {
  return LogicalHelper(*this, LogicalOperator::Neqv, x);
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::DefinedBinary &) {
  Say("TODO: DefinedBinary unimplemented"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::TopLevelChecks(DataRef &&dataRef) {
  if (Component * component{std::get_if<Component>(&dataRef.u)}) {
    CheckUnsubscriptedComponent(*component);
  }
  if (dataRef.Rank() > 0) {
    if (MaybeExpr subscripted{
            ApplySubscripts(std::move(dataRef), std::vector<Subscript>{})}) {
      return subscripted;
    }
  }
  return Designate(std::move(dataRef));
}

void ExprAnalyzer::CheckUnsubscriptedComponent(const Component &component) {
  int baseRank{component.base().Rank()};
  if (baseRank > 0) {
    const Symbol &symbol{component.GetLastSymbol()};
    int componentRank{symbol.Rank()};
    if (componentRank > 0) {
      Say("reference to whole rank-%d component '%%%s' of "
          "rank-%d array of derived type is not allowed"_err_en_US,
          componentRank, symbol.name().ToString().data(), baseRank);
    }
  }
}
}

namespace Fortran::semantics {

class Mutator {
public:
  Mutator(SemanticsContext &context) : context_{context} {}

  template<typename A> bool Pre(A &) { return true /* visit children */; }
  template<typename A> void Post(A &) {}

  bool Pre(parser::Expr &expr) {
    if (expr.typedExpr.get() == nullptr) {
      if (MaybeExpr checked{AnalyzeExpr(context_, expr)}) {
        checked->AsFortran(std::cout << "checked expression: ") << '\n';
        expr.typedExpr.reset(
            new evaluate::GenericExprWrapper{std::move(*checked)});
      } else {
        std::cout << "TODO: expression analysis failed for this expression: ";
        DumpTree(std::cout, expr);
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
};

void AnalyzeExpressions(parser::Program &program, SemanticsContext &context) {
  Mutator mutator{context};
  parser::Walk(program, mutator);
}
}
