// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include "scope.h"
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
#include <optional>

// TODO pmk remove when scaffolding is obsolete
#undef PMKDEBUG
#if PMKDEBUG
#include "dump-parse-tree.h"
#include <iostream>
#endif

// Typedef for optional generic expressions (ubiquitous in this file)
using MaybeExpr =
    std::optional<Fortran::evaluate::Expr<Fortran::evaluate::SomeType>>;

namespace Fortran::parser {
bool SourceLocationFindingVisitor::Pre(const Expr &x) {
  source = x.source;
  return false;
}
void SourceLocationFindingVisitor::Post(const CharBlock &at) { source = at; }
}

// Much of the code that implements semantic analysis of expressions is
// tightly coupled with their typed representations in lib/evaluate,
// and appears here in namespace Fortran::evaluate for convenience.
namespace Fortran::evaluate {

using common::TypeCategory;

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

static std::optional<DataRef> ExtractDataRef(Expr<SomeType> &&expr) {
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

struct DynamicTypeWithLength : public DynamicType {
  std::optional<Expr<SubscriptInteger>> length;
};

std::optional<DynamicTypeWithLength> AnalyzeTypeSpec(
    ExpressionAnalysisContext &context,
    const std::optional<parser::TypeSpec> &spec) {
  if (spec.has_value()) {
    if (const semantics::DeclTypeSpec * typeSpec{spec->declTypeSpec}) {
      // Name resolution sets TypeSpec::declTypeSpec only when it's valid
      // (viz., an intrinsic type with valid known kind or a non-polymorphic
      // & non-ABSTRACT derived type).
      if (const semantics::IntrinsicTypeSpec *
          intrinsic{typeSpec->AsIntrinsic()}) {
        TypeCategory category{intrinsic->category()};
        if (auto kind{ToInt64(intrinsic->kind())}) {
          DynamicTypeWithLength result{category, static_cast<int>(*kind)};
          if (category == TypeCategory::Character) {
            const semantics::CharacterTypeSpec &cts{
                typeSpec->characterTypeSpec()};
            const semantics::ParamValue len{cts.length()};
            // N.B. CHARACTER(LEN=*) is allowed in type-specs in ALLOCATE() &
            // type guards, but not in array constructors.
            if (len.GetExplicit().has_value()) {
              Expr<SomeInteger> copy{*len.GetExplicit()};
              result.length = ConvertToType<SubscriptInteger>(std::move(copy));
            }
          }
          return result;
        }
      } else if (const semantics::DerivedTypeSpec *
          derived{typeSpec->AsDerived()}) {
        return DynamicTypeWithLength{TypeCategory::Derived, 0, derived};
      }
    }
  }
  return std::nullopt;
}

// Forward declarations of additional AnalyzeExpr specializations and overloads
template<typename... As>
MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &, const std::variant<As...> &);
template<typename A>
MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &, const std::optional<A> &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Designator &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::IntLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::SignedIntLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::RealLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::SignedRealLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::ComplexPart &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::ComplexLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::LogicalLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::HollerithLiteralConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::BOZLiteralConstant &);
static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &, const parser::Name &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::NamedConstant &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Substring &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::ArrayElement &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::StructureComponent &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::CoindexedNamedObject &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::CharLiteralConstantSubstring &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::ArrayConstructor &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::StructureConstructor &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::FunctionReference &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Parentheses &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::UnaryPlus &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Negate &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::NOT &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::PercentLoc &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::DefinedUnary &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Power &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Multiply &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Divide &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Add &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Subtract &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::ComplexConstructor &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::Concat &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::LT &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::LE &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::EQ &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::NE &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::GE &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::GT &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::AND &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::OR &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::EQV &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::NEQV &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::XOR &);
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &, const parser::Expr::DefinedBinary &);

// Catch-all unwrapper for AnalyzeExpr's most general case.
template<typename A>
MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context, const A &x) {
  // Some compiler/version/option set combinations used to mysteriously
  // overlook the template specialization in expression.h that
  // redirected parser::Expr arguments, and they would arrive here
  // in the catch-all template.  We've worked around that problem.
  static_assert(
      !std::is_same_v<A, parser::Expr>, "template specialization failed");
  return AnalyzeExpr(context, x.u);
}

// Definitions of AnalyzeExpr() specializations follow.
// Helper subroutines are intermixed.

// Variants and optionals are silently traversed by AnalyzeExpr().
template<typename... As>
MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const std::variant<As...> &u) {
  return std::visit([&](const auto &x) { return AnalyzeExpr(context, x); }, u);
}
template<typename A>
MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const std::optional<A> &x) {
  if (x.has_value()) {
    return AnalyzeExpr(context, *x);
  } else {
    return std::nullopt;
  }
}

// Wraps a object in an explicitly typed representation (e.g., Designator<>
// or FunctionRef<>) that has been instantiated on a dynamically chosen type.
// TODO: move to tools.h?
template<TypeCategory CATEGORY, template<typename> typename WRAPPER,
    typename WRAPPED>
MaybeExpr WrapperHelper(int kind, WRAPPED &&x) {
  return common::SearchTypes(
      TypeKindVisitor<CATEGORY, WRAPPER, WRAPPED>{kind, std::move(x)});
}

template<template<typename> typename WRAPPER, typename WRAPPED>
MaybeExpr TypedWrapper(const DynamicType &dyType, WRAPPED &&x) {
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
  if (std::optional<DynamicType> dyType{GetSymbolType(&symbol)}) {
    return TypedWrapper<Designator, DataRef>(
        std::move(*dyType), std::move(dataRef));
  }
  // TODO: graceful errors on CLASS(*) and TYPE(*) misusage
  return std::nullopt;
}

// Catch and resolve the ambiguous parse of a substring reference
// that looks like a 1-D array element or section.
static MaybeExpr ResolveAmbiguousSubstring(
    ExpressionAnalysisContext &context, ArrayRef &&ref) {
  const Symbol &symbol{ref.GetLastSymbol()};
  if (std::optional<DynamicType> dyType{GetSymbolType(&symbol)}) {
    if (dyType->category == TypeCategory::Character &&
        ref.subscript.size() == 1) {
      DataRef base{std::visit(
          [](auto &&y) { return DataRef{std::move(y)}; }, std::move(ref.u))};
      std::optional<Expr<SubscriptInteger>> lower, upper;
      if (std::visit(
              common::visitors{
                  [&](IndirectSubscriptIntegerExpr &&x) {
                    lower = std::move(*x);
                    return true;
                  },
                  [&](Triplet &&triplet) {
                    lower = triplet.lower();
                    upper = triplet.upper();
                    return triplet.IsStrideOne();
                  },
              },
              std::move(ref.subscript[0].u))) {
        return WrapperHelper<TypeCategory::Character, Designator, Substring>(
            dyType->kind,
            Substring{std::move(base), std::move(lower), std::move(upper)});
      }
    }
  }

  return std::nullopt;
}

// Some subscript semantic checks must be deferred until all of the
// subscripts are in hand.  This is also where we can catch the
// ambiguous parse of a substring reference that looks like a 1-D array
// element or section.
static MaybeExpr CompleteSubscripts(
    ExpressionAnalysisContext &context, ArrayRef &&ref) {
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
    if (MaybeExpr substring{
            ResolveAmbiguousSubstring(context, std::move(ref))}) {
      return substring;
    }
    context.Say("Reference to rank-%d object '%s' has %d subscripts"_err_en_US,
        symbolRank, symbol.name().ToString().data(), subscripts);
  } else if (subscripts == 0) {
    // nothing to check
  } else if (Component * component{std::get_if<Component>(&ref.u)}) {
    int baseRank{component->Rank()};
    if (baseRank > 0) {
      int rank{ref.Rank()};
      if (rank > 0) {
        context.Say("Subscripts of rank-%d component reference have rank %d, "
                    "but must all be scalar"_err_en_US,
            baseRank, rank);
      }
    }
  } else if (const auto *details{
                 symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    // C928 & C1002
    if (Triplet * last{std::get_if<Triplet>(&ref.subscript.back().u)}) {
      if (!last->upper().has_value() && details->IsAssumedSize()) {
        context.Say("Assumed-size array '%s' must have explicit final "
                    "subscript upper bound value"_err_en_US,
            symbol.name().ToString().data());
      }
    }
  }
  return Designate(DataRef{std::move(ref)});
}

// Applies subscripts to a data reference.
static MaybeExpr ApplySubscripts(ExpressionAnalysisContext &context,
    DataRef &&dataRef, std::vector<Subscript> &&subscripts) {
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) {
            return CompleteSubscripts(
                context, ArrayRef{*symbol, std::move(subscripts)});
          },
          [&](auto &&base) -> MaybeExpr {
            using Ty = std::decay_t<decltype(base)>;
            if constexpr (common::HasMember<Ty, decltype(ArrayRef::u)>) {
              return CompleteSubscripts(
                  context, ArrayRef{std::move(base), std::move(subscripts)});
            }
            return std::nullopt;
          },
      },
      std::move(dataRef.u));
}

// Ensure that a whole component reference made to an array of derived type
// does not also reference an array (e.g., A(:)%ARRAY is invalid).
static void CheckUnsubscriptedComponent(
    ExpressionAnalysisContext &context, const Component &component) {
  int baseRank{component.base().Rank()};
  if (baseRank > 0) {
    const Symbol &symbol{component.GetLastSymbol()};
    int componentRank{symbol.Rank()};
    if (componentRank > 0) {
      context.Say("reference to whole rank-%d component '%%%s' of "
                  "rank-%d array of derived type is not allowed"_err_en_US,
          componentRank, symbol.name().ToString().data(), baseRank);
    }
  }
}

// Top-level checks for data references.  Unsubscripted whole array references
// get expanded -- e.g., MATRIX becomes MATRIX(:,:).
static MaybeExpr TopLevelChecks(
    ExpressionAnalysisContext &context, DataRef &&dataRef) {
  if (Component * component{std::get_if<Component>(&dataRef.u)}) {
    CheckUnsubscriptedComponent(context, *component);
  }
  if (dataRef.Rank() > 0) {
    if (MaybeExpr subscripted{ApplySubscripts(
            context, std::move(dataRef), std::vector<Subscript>{})}) {
      return subscripted;
    }
  }
  return Designate(std::move(dataRef));
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Designator &d) {
  // These checks have to be deferred to these "top level" data-refs where
  // we can be sure that there are no following subscripts (yet).
  if (MaybeExpr result{AnalyzeExpr(context, d.u)}) {
    if (std::optional<evaluate::DataRef> dataRef{
            evaluate::ExtractDataRef(std::move(result))}) {
      return TopLevelChecks(context, std::move(*dataRef));
    }
    return result;
  }
  return std::nullopt;
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

// Type kind parameter values for literal constants.
static int AnalyzeKindParam(ExpressionAnalysisContext &context,
    const std::optional<parser::KindParam> &kindParam, int defaultKind,
    int kanjiKind = -1) {
  if (!kindParam.has_value()) {
    return defaultKind;
  }
  return std::visit(
      common::visitors{
          [](std::uint64_t k) { return static_cast<int>(k); },
          [&](const parser::Scalar<
              parser::Integer<parser::Constant<parser::Name>>> &n) {
            if (MaybeExpr ie{AnalyzeExpr(context, n)}) {
              if (std::optional<std::int64_t> i64{ToInt64(*ie)}) {
                int iv = *i64;
                if (iv == *i64) {
                  return iv;
                }
              }
            }
            context.Say("KIND type parameter on literal must be a scalar "
                        "integer constant"_err_en_US);
            return defaultKind;
          },
          [&](parser::KindParam::Kanji) {
            if (kanjiKind >= 0) {
              return kanjiKind;
            }
            context.Say("Kanji not allowed here"_err_en_US);
            return defaultKind;
          },
      },
      kindParam->u);
}

// Common handling of parser::IntLiteralConstant and SignedIntLiteralConstant
template<typename PARSED>
MaybeExpr IntLiteralConstant(
    ExpressionAnalysisContext &context, const PARSED &x) {
  int kind{
      AnalyzeKindParam(context, std::get<std::optional<parser::KindParam>>(x.t),
          context.GetDefaultKind(TypeCategory::Integer))};
  auto value{std::get<0>(x.t)};  // std::(u)int64_t
  auto result{common::SearchTypes(
      TypeKindVisitor<TypeCategory::Integer, Constant, std::int64_t>{
          kind, static_cast<std::int64_t>(value)})};
  if (!result.has_value()) {
    context.Say("unsupported INTEGER(KIND=%d)"_err_en_US, kind);
  }
  return result;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::IntLiteralConstant &x) {
  return IntLiteralConstant(context, x);
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::SignedIntLiteralConstant &x) {
  return IntLiteralConstant(context, x);
}

template<typename TYPE>
Constant<TYPE> ReadRealLiteral(
    parser::CharBlock source, FoldingContext &context) {
  const char *p{source.begin()};
  auto valWithFlags{Scalar<TYPE>::Read(p, context.rounding)};
  CHECK(p == source.end());
  RealFlagWarnings(context, valWithFlags.flags, "conversion of REAL literal");
  auto value{valWithFlags.value};
  if (context.flushSubnormalsToZero) {
    value = value.FlushSubnormalToZero();
  }
  return {value};
}

struct RealTypeVisitor {
  using Result = std::optional<Expr<SomeReal>>;
  using Types = RealTypes;

  RealTypeVisitor(int k, parser::CharBlock lit, FoldingContext &ctx)
    : kind{k}, literal{lit}, context{ctx} {}

  template<typename T> Result Test() {
    if (kind == T::kind) {
      return {AsCategoryExpr(ReadRealLiteral<T>(literal, context))};
    }
    return std::nullopt;
  }

  int kind;
  parser::CharBlock literal;
  FoldingContext &context;
};

// Reads a real literal constant and encodes it with the right kind.
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::RealLiteralConstant &x) {
  // Use a local message context around the real literal for better
  // provenance on any messages.
  auto save{context.GetContextualMessages().SetLocation(x.real.source)};
  // If a kind parameter appears, it defines the kind of the literal and any
  // letter used in an exponent part (e.g., the 'E' in "6.02214E+23")
  // should agree.  In the absence of an explicit kind parameter, any exponent
  // letter determines the kind.  Otherwise, defaults apply.
  // TODO: warn on inexact conversions?
  auto &defaults{context.context().defaultKinds()};
  int defaultKind{defaults.GetDefaultKind(TypeCategory::Real)};
  const char *end{x.real.source.end()};
  std::optional<int> letterKind;
  for (const char *p{x.real.source.begin()}; p < end; ++p) {
    if (parser::IsLetter(*p)) {
      switch (*p) {
      case 'e': letterKind = defaults.GetDefaultKind(TypeCategory::Real); break;
      case 'd': letterKind = defaults.doublePrecisionKind(); break;
      case 'q': letterKind = defaults.quadPrecisionKind(); break;
      default: context.Say("unknown exponent letter '%c'"_err_en_US, *p);
      }
      break;
    }
  }
  if (letterKind.has_value()) {
    defaultKind = *letterKind;
  }
  auto kind{AnalyzeKindParam(context, x.kind, defaultKind)};
  if (letterKind.has_value() && kind != *letterKind) {
    context.Say(
        "explicit kind parameter on real constant disagrees with exponent letter"_en_US);
  }
  auto result{common::SearchTypes(
      RealTypeVisitor{kind, x.real.source, context.GetFoldingContext()})};
  if (!result.has_value()) {
    context.Say("unsupported REAL(KIND=%d)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::SignedRealLiteralConstant &x) {
  if (MaybeExpr result{
          AnalyzeExpr(context, std::get<parser::RealLiteralConstant>(x.t))}) {
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

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::ComplexPart &x) {
  return AnalyzeExpr(context, x.u);
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::ComplexLiteralConstant &z) {
  return AsMaybeExpr(ConstructComplex(context.GetContextualMessages(),
      AnalyzeExpr(context, std::get<0>(z.t)),
      AnalyzeExpr(context, std::get<1>(z.t)),
      context.GetDefaultKind(TypeCategory::Real)));
}

// CHARACTER literal processing.
static MaybeExpr AnalyzeString(
    ExpressionAnalysisContext &context, std::string &&string, int kind) {
  if (!IsValidKindOfIntrinsicType(TypeCategory::Character, kind)) {
    context.Say("unsupported CHARACTER(KIND=%d)"_err_en_US, kind);
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
    context.Say(
        "bad UTF-8 encoding of CHARACTER(KIND=%d) literal"_err_en_US, kind);
    return std::nullopt;
  }
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::CharLiteralConstant &x) {
  int kind{AnalyzeKindParam(
      context, std::get<std::optional<parser::KindParam>>(x.t), 1)};
  auto value{std::get<std::string>(x.t)};
  return AnalyzeString(context, std::move(value), kind);
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::HollerithLiteralConstant &x) {
  int kind{context.GetDefaultKind(TypeCategory::Character)};
  auto value{x.v};
  return AnalyzeString(context, std::move(value), kind);
}

// .TRUE. and .FALSE. of various kinds
static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::LogicalLiteralConstant &x) {
  auto kind{
      AnalyzeKindParam(context, std::get<std::optional<parser::KindParam>>(x.t),
          context.GetDefaultKind(TypeCategory::Logical))};
  bool value{std::get<bool>(x.t)};
  auto result{common::SearchTypes(
      TypeKindVisitor<TypeCategory::Logical, Constant, bool>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    context.Say("unsupported LOGICAL(KIND=%d)"_err_en_US, kind);
  }
  return result;
}

// BOZ typeless literals
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::BOZLiteralConstant &x) {
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
    context.Say(
        "invalid digit ('%c') in BOZ literal %s"_err_en_US, *p, x.v.data());
    return std::nullopt;
  }
  if (value.overflow) {
    context.Say("BOZ literal %s too large"_err_en_US, x.v.data());
    return std::nullopt;
  }
  return {AsGenericExpr(std::move(value.value))};
}

// For use with SearchTypes to create a TypeParamInquiry with the
// right integer kind.
struct TypeParamInquiryVisitor {
  using Result = std::optional<Expr<SomeInteger>>;
  using Types = IntegerTypes;
  TypeParamInquiryVisitor(int k, SymbolOrComponent &&b, const Symbol &param)
    : kind{k}, base{std::move(b)}, parameter{param} {}
  template<typename T> Result Test() {
    if (kind == T::kind) {
      return Expr<SomeInteger>{
          Expr<T>{TypeParamInquiry<T::kind>{std::move(base), parameter}}};
    }
    return std::nullopt;
  }
  int kind;
  SymbolOrComponent base;
  const Symbol &parameter;
};

static std::optional<Expr<SomeInteger>> MakeTypeParamInquiry(
    const Symbol *symbol) {
  if (std::optional<DynamicType> dyType{GetSymbolType(symbol)}) {
    if (dyType->category == TypeCategory::Integer) {
      return common::SearchTypes(TypeParamInquiryVisitor{
          dyType->kind, SymbolOrComponent{nullptr}, *symbol});
    }
  }
  return std::nullopt;
}

// Names and named constants
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Name &n) {
  if (std::optional<int> kind{context.IsAcImpliedDo(n.source)}) {
    return AsMaybeExpr(ConvertToKind<TypeCategory::Integer>(
        *kind, AsExpr(ImpliedDoIndex{n.source})));
  } else if (n.symbol == nullptr) {
    context.Say(
        n.source, "TODO INTERNAL: name was not resolved to a symbol"_err_en_US);
  } else if (n.symbol->attrs().test(semantics::Attr::PARAMETER)) {
    if (auto *details{n.symbol->detailsIf<semantics::ObjectEntityDetails>()}) {
      if (auto &init{details->init()}) {
        return init;
      }
    }
    context.Say(n.source, "parameter does not have a value"_err_en_US);
    // TODO: enumerators, do they have the PARAMETER attribute?
  } else if (n.symbol->detailsIf<semantics::TypeParamDetails>()) {
    // A bare reference to a derived type parameter (within a parameterized
    // derived type definition)
    return AsMaybeExpr(MakeTypeParamInquiry(n.symbol));
  } else if (MaybeExpr result{Designate(DataRef{*n.symbol})}) {
    return result;
  } else {
    context.Say(n.source, "not of a supported type and kind"_err_en_US);
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::NamedConstant &n) {
  if (MaybeExpr value{AnalyzeExpr(context, n.v)}) {
    Expr<SomeType> folded{Fold(context.GetFoldingContext(), std::move(*value))};
    if (IsConstantExpr(folded)) {
      return {folded};
    }
    context.Say(n.v.source, "must be a constant"_err_en_US);
  }
  return std::nullopt;
}

// Substring references
static std::optional<Expr<SubscriptInteger>> GetSubstringBound(
    ExpressionAnalysisContext &context,
    const std::optional<parser::ScalarIntExpr> &bound) {
  if (bound.has_value()) {
    if (MaybeExpr expr{AnalyzeExpr(context, *bound)}) {
      if (expr->Rank() > 1) {
        context.Say(
            "substring bound expression has rank %d"_err_en_US, expr->Rank());
      }
      if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
        if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
          return {std::move(*ssIntExpr)};
        }
        return {Expr<SubscriptInteger>{
            Convert<SubscriptInteger, TypeCategory::Integer>{
                std::move(*intExpr)}}};
      } else {
        context.Say("substring bound expression is not INTEGER"_err_en_US);
      }
    }
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Substring &ss) {
  if (MaybeExpr baseExpr{
          AnalyzeExpr(context, std::get<parser::DataRef>(ss.t))}) {
    if (std::optional<DataRef> dataRef{ExtractDataRef(std::move(*baseExpr))}) {
      if (MaybeExpr newBaseExpr{TopLevelChecks(context, std::move(*dataRef))}) {
        if (std::optional<DataRef> checked{
                ExtractDataRef(std::move(*newBaseExpr))}) {
          const parser::SubstringRange &range{
              std::get<parser::SubstringRange>(ss.t)};
          std::optional<Expr<SubscriptInteger>> first{
              GetSubstringBound(context, std::get<0>(range.t))};
          std::optional<Expr<SubscriptInteger>> last{
              GetSubstringBound(context, std::get<1>(range.t))};
          const Symbol &symbol{checked->GetLastSymbol()};
          if (std::optional<DynamicType> dynamicType{GetSymbolType(&symbol)}) {
            if (dynamicType->category == TypeCategory::Character) {
              return WrapperHelper<TypeCategory::Character, Designator,
                  Substring>(dynamicType->kind,
                  Substring{std::move(checked.value()), std::move(first),
                      std::move(last)});
            }
          }
          context.Say("substring may apply only to CHARACTER"_err_en_US);
        }
      }
    }
  }
  return std::nullopt;
}

// CHARACTER literal substrings
static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::CharLiteralConstantSubstring &x) {
  const parser::SubstringRange &range{std::get<parser::SubstringRange>(x.t)};
  std::optional<Expr<SubscriptInteger>> lower{
      GetSubstringBound(context, std::get<0>(range.t))};
  std::optional<Expr<SubscriptInteger>> upper{
      GetSubstringBound(context, std::get<1>(range.t))};
  if (MaybeExpr string{
          AnalyzeExpr(context, std::get<parser::CharLiteralConstant>(x.t))}) {
    if (auto *charExpr{std::get_if<Expr<SomeCharacter>>(&string->u)}) {
      Expr<SubscriptInteger> length{std::visit(
          [](const auto &ckExpr) { return ckExpr.LEN(); }, charExpr->u)};
      if (!lower.has_value()) {
        lower = Expr<SubscriptInteger>{1};
      }
      if (!upper.has_value()) {
        upper = Expr<SubscriptInteger>{
            static_cast<std::int64_t>(ToInt64(length).value())};
      }
      return std::visit(
          [&](auto &&ckExpr) -> MaybeExpr {
            using Result = ResultType<decltype(ckExpr)>;
            auto *cp{std::get_if<Constant<Result>>(&ckExpr.u)};
            CHECK(cp != nullptr);  // the parent was parsed as a constant string
            CHECK(cp->size() == 1);
            StaticDataObject::Pointer staticData{StaticDataObject::Create()};
            staticData->set_alignment(Result::kind)
                .set_itemBytes(Result::kind)
                .Push(**cp);
            Substring substring{std::move(staticData), std::move(lower.value()),
                std::move(upper.value())};
            return AsGenericExpr(Expr<SomeCharacter>{
                Expr<Result>{Designator<Result>{std::move(substring)}}});
          },
          std::move(charExpr->u));
    }
  }
  return std::nullopt;
}

// Subscripted array references
static std::optional<Expr<SubscriptInteger>> AsSubscript(
    ExpressionAnalysisContext &context, MaybeExpr &&expr) {
  if (expr.has_value()) {
    if (expr->Rank() > 1) {
      context.Say("subscript expression has rank %d"_err_en_US, expr->Rank());
    }
    if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
      if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
        return {std::move(*ssIntExpr)};
      }
      return {Expr<SubscriptInteger>{
          Convert<SubscriptInteger, TypeCategory::Integer>{
              std::move(*intExpr)}}};
    } else {
      context.Say("subscript expression is not INTEGER"_err_en_US);
    }
  }
  return std::nullopt;
}

static std::optional<Expr<SubscriptInteger>> TripletPart(
    ExpressionAnalysisContext &context,
    const std::optional<parser::Subscript> &s) {
  if (s.has_value()) {
    return AsSubscript(context, AnalyzeExpr(context, *s));
  }
  return std::nullopt;
}

static std::optional<Subscript> AnalyzeSectionSubscript(
    ExpressionAnalysisContext &context, const parser::SectionSubscript &ss) {
  return std::visit(
      common::visitors{
          [&](const parser::SubscriptTriplet &t) {
            return std::make_optional(
                Subscript{Triplet{TripletPart(context, std::get<0>(t.t)),
                    TripletPart(context, std::get<1>(t.t)),
                    TripletPart(context, std::get<2>(t.t))}});
          },
          [&](const auto &s) -> std::optional<Subscript> {
            if (auto subscriptExpr{
                    AsSubscript(context, AnalyzeExpr(context, s))}) {
              return {Subscript{std::move(*subscriptExpr)}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

static std::vector<Subscript> AnalyzeSectionSubscripts(
    ExpressionAnalysisContext &context,
    const std::list<parser::SectionSubscript> &sss) {
  std::vector<Subscript> subscripts;
  for (const auto &s : sss) {
    if (auto subscript{AnalyzeSectionSubscript(context, s)}) {
      subscripts.emplace_back(std::move(*subscript));
    }
  }
  return subscripts;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::ArrayElement &ae) {
  std::vector<Subscript> subscripts{
      AnalyzeSectionSubscripts(context, ae.subscripts)};
  if (MaybeExpr baseExpr{AnalyzeExpr(context, ae.base)}) {
    if (std::optional<DataRef> dataRef{ExtractDataRef(std::move(*baseExpr))}) {
      if (MaybeExpr result{ApplySubscripts(
              context, std::move(*dataRef), std::move(subscripts))}) {
        return result;
      }
    }
  }
  context.Say(
      "subscripts may be applied only to an object or component"_err_en_US);
  return std::nullopt;
}

// Type parameter inquiries apply to data references, but don't depend
// on any trailing (co)subscripts.
static SymbolOrComponent IgnoreAnySubscripts(
    Designator<SomeDerived> &&designator) {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return SymbolOrComponent{symbol}; },
          [](Component &&component) { return SymbolOrComponent{component}; },
          [](ArrayRef &&arrayRef) { return std::move(arrayRef.u); },
          [](CoarrayRef &&coarrayRef) {
            return SymbolOrComponent{&coarrayRef.GetLastSymbol()};
          },
      },
      std::move(designator.u));
}

// Components of parent derived types are explicitly represented as such.
static std::optional<Component> CreateComponent(
    DataRef &&base, const Symbol &component, const semantics::Scope &scope) {
  if (&component.owner() == &scope) {
    return {Component{std::move(base), component}};
  }
  if (const semantics::Scope * parentScope{scope.GetDerivedTypeParent()}) {
    if (const Symbol * parentComponent{parentScope->GetSymbol()}) {
      return CreateComponent(
          DataRef{Component{std::move(base), *parentComponent}}, component,
          *parentScope);
    }
  }
  return std::nullopt;
}

// Derived type component references and type parameter inquiries
static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::StructureComponent &sc) {
  const auto &name{sc.component.source};
  if (MaybeExpr base{AnalyzeExpr(context, sc.base)}) {
    Symbol *sym{sc.component.symbol};
    if (sym == nullptr) {
      context.Say(sc.component.source,
          "component name was not resolved to a symbol"_err_en_US);
    } else if (auto *dtExpr{UnwrapExpr<Expr<SomeDerived>>(*base)}) {
      const semantics::DerivedTypeSpec *dtSpec{nullptr};
      if (std::optional<DynamicType> dtDyTy{dtExpr->GetType()}) {
        dtSpec = dtDyTy->derived;
      }
      if (sym->detailsIf<semantics::TypeParamDetails>()) {
        if (auto *designator{UnwrapExpr<Designator<SomeDerived>>(*dtExpr)}) {
          std::optional<DynamicType> dyType{GetSymbolType(sym)};
          CHECK(dyType.has_value());
          CHECK(dyType->category == TypeCategory::Integer);
          return AsMaybeExpr(
              common::SearchTypes(TypeParamInquiryVisitor{dyType->kind,
                  IgnoreAnySubscripts(std::move(*designator)), *sym}));
        } else {
          context.Say(name,
              "type parameter inquiry must be applied to a designator"_err_en_US);
        }
      } else if (dtSpec == nullptr || dtSpec->scope() == nullptr) {
        context.Say(name,
            "TODO: base of component reference lacks a derived type"_err_en_US);
      } else if (std::optional<DataRef> dataRef{
                     ExtractDataRef(std::move(*dtExpr))}) {
        if (auto component{
                CreateComponent(std::move(*dataRef), *sym, *dtSpec->scope())}) {
          return Designate(DataRef{std::move(*component)});
        } else {
          context.Say(name,
              "component is not in scope of derived TYPE(%s)"_err_en_US,
              dtSpec->typeSymbol().name().ToString().data());
        }
      } else {
        context.Say(name,
            "base of component reference must be a data reference"_err_en_US);
      }
    } else if (auto *details{sym->detailsIf<semantics::MiscDetails>()}) {
      // special part-ref: %re, %im, %kind, %len
      // Type errors are detected and reported in semantics.
      using MiscKind = semantics::MiscDetails::Kind;
      MiscKind kind{details->kind()};
      if (kind == MiscKind::ComplexPartRe || kind == MiscKind::ComplexPartIm) {
        if (auto *zExpr{std::get_if<Expr<SomeComplex>>(&base->u)}) {
          if (std::optional<DataRef> dataRef{
                  ExtractDataRef(std::move(*zExpr))}) {
            Expr<SomeReal> realExpr{std::visit(
                [&](const auto &z) {
                  using PartType = typename ResultType<decltype(z)>::Part;
                  auto part{kind == MiscKind::ComplexPartRe
                          ? ComplexPart::Part::RE
                          : ComplexPart::Part::IM};
                  return AsCategoryExpr(Designator<PartType>{
                      ComplexPart{std::move(*dataRef), part}});
                },
                zExpr->u)};
            return {AsGenericExpr(std::move(realExpr))};
          }
        }
      } else if (kind == MiscKind::KindParamInquiry ||
          kind == MiscKind::LenParamInquiry) {
        // Convert x%KIND -> intrinsic KIND(x), x%LEN -> intrinsic LEN(x)
        SpecificIntrinsic func{name.ToString()};
        func.type = context.GetDefaultKindOfType(TypeCategory::Integer);
        return TypedWrapper<FunctionRef, ProcedureRef>(*func.type,
            ProcedureRef{ProcedureDesignator{std::move(func)},
                ActualArguments{ActualArgument{std::move(*base)}}});
      } else {
        common::die("unexpected kind");
      }
    } else {
      context.Say(
          name, "derived type required before component reference"_err_en_US);
    }
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::CoindexedNamedObject &co) {
  // TODO: CheckUnsubscriptedComponent or its equivalent
  context.Say("TODO: CoindexedNamedObject unimplemented"_err_en_US);
  return std::nullopt;
}

static int IntegerTypeSpecKind(
    ExpressionAnalysisContext &context, const parser::IntegerTypeSpec &spec) {
  Expr<SubscriptInteger> value{context.Analyze(TypeCategory::Integer, spec.v)};
  if (auto kind{ToInt64(value)}) {
    return static_cast<int>(*kind);
  }
  context.SayAt(spec, "Constant INTEGER kind value required here"_err_en_US);
  return context.GetDefaultKind(TypeCategory::Integer);
}

template<int KIND, typename A>
std::optional<Expr<Type<TypeCategory::Integer, KIND>>> GetSpecificIntExpr(
    ExpressionAnalysisContext &context, const A &x) {
  if (MaybeExpr y{AnalyzeExpr(context, x)}) {
    Expr<SomeInteger> *intExpr{UnwrapExpr<Expr<SomeInteger>>(*y)};
    CHECK(intExpr != nullptr);
    return ConvertToType<Type<TypeCategory::Integer, KIND>>(
        std::move(*intExpr));
  }
  return std::nullopt;
}

// Array constructors

struct ArrayConstructorContext {
  void Push(MaybeExpr &&);
  void Add(const parser::AcValue &);
  ExpressionAnalysisContext &exprContext;
  std::optional<DynamicTypeWithLength> &type;
  bool typesMustMatch{false};
  ArrayConstructorValues<SomeType> values;
};

void ArrayConstructorContext::Push(MaybeExpr &&x) {
  if (x.has_value()) {
    DynamicTypeWithLength xType;
    if (auto dyType{x->GetType()}) {
      *static_cast<DynamicType *>(&xType) = *dyType;
    }
    if (Expr<SomeCharacter> * charExpr{UnwrapExpr<Expr<SomeCharacter>>(*x)}) {
      CHECK(xType.category == TypeCategory::Character);
      xType.length =
          std::visit([](const auto &kc) { return kc.LEN(); }, charExpr->u);
    }
    if (!type.has_value()) {
      // If there is no explicit type-spec in an array constructor, the type
      // of the array is the declared type of all of the elements, which must
      // be well-defined.
      // TODO: Possible language extension: use the most general type of
      // the values as the type of a numeric constructed array, convert all
      // of the other values to that type.  Alternative: let the first value
      // determine the type, and convert the others to that type.
      type = std::move(xType);
      values.Push(std::move(*x));
    } else if (typesMustMatch) {
      if (static_cast<const DynamicType &>(*type) ==
          static_cast<const DynamicType &>(xType)) {
        values.Push(std::move(*x));
      } else {
        exprContext.Say(
            "Values in array constructor must have the same declared type when no explicit type appears"_err_en_US);
      }
    } else {
      if (auto cast{ConvertToType(*type, std::move(*x))}) {
        values.Push(std::move(*cast));
      } else {
        exprContext.Say(
            "Value in array constructor could not be converted to the type of the array"_err_en_US);
      }
    }
  }
}

void ArrayConstructorContext::Add(const parser::AcValue &x) {
  using IntType = ResultType<ImpliedDoIndex>;
  std::visit(
      common::visitors{
          [&](const parser::AcValue::Triplet &triplet) {
            // Transform l:u(:s) into (_,_=l,u(,s)) with an anonymous index '_'
            std::optional<Expr<IntType>> lower{
                GetSpecificIntExpr<IntType::kind>(
                    exprContext, std::get<0>(triplet.t))};
            std::optional<Expr<IntType>> upper{
                GetSpecificIntExpr<IntType::kind>(
                    exprContext, std::get<1>(triplet.t))};
            std::optional<Expr<IntType>> stride{
                GetSpecificIntExpr<IntType::kind>(
                    exprContext, std::get<2>(triplet.t))};
            if (lower.has_value() && upper.has_value()) {
              if (!stride.has_value()) {
                stride = Expr<IntType>{1};
              }
              if (!type.has_value()) {
                type = DynamicTypeWithLength{IntType::GetType()};
              }
              ArrayConstructorContext nested{exprContext, type, typesMustMatch};
              parser::CharBlock name;
              nested.Push(Expr<SomeType>{
                  Expr<SomeInteger>{Expr<IntType>{ImpliedDoIndex{name}}}});
              values.Push(ImpliedDo<SomeType>{name, std::move(*lower),
                  std::move(*upper), std::move(*stride),
                  std::move(nested.values)});
            }
          },
          [&](const common::Indirection<parser::Expr> &expr) {
            if (MaybeExpr v{exprContext.Analyze(*expr)}) {
              Push(std::move(*v));
            }
          },
          [&](const common::Indirection<parser::AcImpliedDo> &impliedDo) {
            const auto &control{
                std::get<parser::AcImpliedDoControl>(impliedDo->t)};
            const auto &bounds{
                std::get<parser::LoopBounds<parser::ScalarIntExpr>>(control.t)};
            parser::CharBlock name{bounds.name.thing.thing.source};
            int kind{IntType::kind};
            if (auto &its{std::get<std::optional<parser::IntegerTypeSpec>>(
                    control.t)}) {
              kind = IntegerTypeSpecKind(exprContext, *its);
            }
            bool inserted{exprContext.AddAcImpliedDo(name, kind)};
            if (!inserted) {
              exprContext.SayAt(name,
                  "Implied DO index is active in surrounding implied DO loop and cannot have the same name"_err_en_US);
            }
            std::optional<Expr<IntType>> lower{
                GetSpecificIntExpr<IntType::kind>(exprContext, bounds.lower)};
            std::optional<Expr<IntType>> upper{
                GetSpecificIntExpr<IntType::kind>(exprContext, bounds.upper)};
            std::optional<Expr<IntType>> stride{
                GetSpecificIntExpr<IntType::kind>(exprContext, bounds.step)};
            ArrayConstructorContext nested{exprContext, type, typesMustMatch};
            for (const auto &value :
                std::get<std::list<parser::AcValue>>(impliedDo->t)) {
              nested.Add(value);
            }
            if (lower.has_value() && upper.has_value()) {
              if (!stride.has_value()) {
                stride = Expr<IntType>{1};
              }
              values.Push(ImpliedDo<SomeType>{name, std::move(*lower),
                  std::move(*upper), std::move(*stride),
                  std::move(nested.values)});
            }
            if (inserted) {
              exprContext.RemoveAcImpliedDo(name);
            }
          },
      },
      x.u);
}

// Inverts a collection of generic ArrayConstructorValues<SomeType> that
// all happen to have or be convertible to the same actual type T into
// one ArrayConstructor<T>.
template<typename T>
ArrayConstructorValues<T> MakeSpecific(
    ArrayConstructorValues<SomeType> &&from) {
  ArrayConstructorValues<T> to;
  for (ArrayConstructorValue<SomeType> &x : from.values) {
    std::visit(
        common::visitors{
            [&](CopyableIndirection<Expr<SomeType>> &&expr) {
              auto *typed{UnwrapExpr<Expr<T>>(*expr)};
              CHECK(typed != nullptr);
              to.Push(std::move(*typed));
            },
            [&](ImpliedDo<SomeType> &&impliedDo) {
              to.Push(ImpliedDo<T>{impliedDo.controlVariableName,
                  std::move(*impliedDo.lower), std::move(*impliedDo.upper),
                  std::move(*impliedDo.stride),
                  MakeSpecific<T>(std::move(*impliedDo.values))});
            },
        },
        std::move(x.u));
  }
  return to;
}

struct ArrayConstructorTypeVisitor {
  using Result = MaybeExpr;
  using Types = LengthlessIntrinsicTypes;
  template<typename T> Result Test() {
    if (type.category == T::category && type.kind == T::kind) {
      if constexpr (T::category == TypeCategory::Character) {
        CHECK(type.length.has_value());
        return AsMaybeExpr(ArrayConstructor<T>{
            MakeSpecific<T>(std::move(values)), std::move(*type.length)});
      } else {
        return AsMaybeExpr(
            ArrayConstructor<T>{T{}, MakeSpecific<T>(std::move(values))});
      }
    } else {
      return std::nullopt;
    }
  }
  DynamicTypeWithLength type;
  ArrayConstructorValues<SomeType> values;
};

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &exprContext,
    const parser::ArrayConstructor &array) {
  const parser::AcSpec &acSpec{array.v};
  std::optional<DynamicTypeWithLength> type{
      AnalyzeTypeSpec(exprContext, acSpec.type)};
  bool typesMustMatch{!type.has_value()};
  ArrayConstructorContext context{exprContext, type, typesMustMatch};
  for (const parser::AcValue &value : acSpec.values) {
    context.Add(value);
  }
  if (type.has_value()) {
    ArrayConstructorTypeVisitor visitor{
        std::move(*type), std::move(context.values)};
    return common::SearchTypes(std::move(visitor));
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::StructureConstructor &) {
  context.Say("TODO: StructureConstructor unimplemented"_en_US);
  return std::nullopt;
}

static std::optional<CallAndArguments> Procedure(
    ExpressionAnalysisContext &context, const parser::ProcedureDesignator &pd,
    ActualArguments &arguments) {
  return std::visit(
      common::visitors{
          [&](const parser::Name &n) -> std::optional<CallAndArguments> {
            if (n.symbol == nullptr) {
              context.Say(
                  "TODO INTERNAL no symbol for procedure designator name '%s'"_err_en_US,
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
                                context.context().intrinsics().Probe(cc,
                                    arguments,
                                    &context.GetContextualMessages())}) {
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
                      context.Say(
                          "TODO: unimplemented/invalid kind of symbol as procedure designator '%s'"_err_en_US,
                          n.ToString().data());
                      return std::nullopt;
                    },
                },
                n.symbol->details());
          },
          [&](const parser::ProcComponentRef &pcr)
              -> std::optional<CallAndArguments> {
            if (MaybeExpr component{AnalyzeExpr(context, pcr.v)}) {
              // TODO distinguish PCR from TBP
              // TODO optional PASS argument for TBP
              context.Say("TODO: proc component ref"_err_en_US);
              return std::nullopt;
            } else {
              return std::nullopt;
            }
          },
      },
      pd.u);
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::FunctionReference &funcRef) {
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
              actualArgExpr = AnalyzeExpr(context, *v);
            },
            [&](const common::Indirection<parser::Expr> &x) {
              actualArgExpr = AnalyzeExpr(context, *x);
            },
            [&](const parser::Name &n) {
              context.Say("TODO: procedure name actual arg"_err_en_US);
            },
            [&](const parser::ProcComponentRef &) {
              context.Say("TODO: proc component ref actual arg"_err_en_US);
            },
            [&](const parser::AltReturnSpec &) {
              context.Say(
                  "alternate return specification cannot appear on function reference"_err_en_US);
            },
            [&](const parser::ActualArg::PercentRef &) {
              context.Say("TODO: %REF() argument"_err_en_US);
            },
            [&](const parser::ActualArg::PercentVal &) {
              context.Say("TODO: %VAL() argument"_err_en_US);
            },
        },
        std::get<parser::ActualArg>(arg.t).u);
    if (actualArgExpr.has_value()) {
      arguments.emplace_back(std::make_optional(
          Fold(context.GetFoldingContext(), std::move(*actualArgExpr))));
      if (const auto &argKW{std::get<std::optional<parser::Keyword>>(arg.t)}) {
        arguments.back()->keyword = argKW->v.source;
      }
    } else {
      return std::nullopt;
    }
  }

  // TODO: map user generic to specific procedure
  if (std::optional<CallAndArguments> proc{Procedure(context,
          std::get<parser::ProcedureDesignator>(funcRef.v.t), arguments)}) {
    if (std::optional<DynamicType> dyType{
            proc->procedureDesignator.GetType()}) {
      return TypedWrapper<FunctionRef, ProcedureRef>(*dyType,
          ProcedureRef{std::move(proc->procedureDesignator),
              std::move(proc->arguments)});
    }
  }
  return std::nullopt;
}

// Unary operations

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Parentheses &x) {
  // TODO: C1003: A parenthesized function reference may not return a
  // procedure pointer.
  if (MaybeExpr operand{AnalyzeExpr(context, *x.v)}) {
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

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::UnaryPlus &x) {
  MaybeExpr value{AnalyzeExpr(context, *x.v)};
  if (value.has_value()) {
    std::visit(
        common::visitors{
            [](const BOZLiteralConstant &) {},  // allow +Z'1', it's harmless
            [&](const auto &catExpr) {
              TypeCategory cat{ResultType<decltype(catExpr)>::category};
              if (cat != TypeCategory::Integer && cat != TypeCategory::Real &&
                  cat != TypeCategory::Complex) {
                context.Say(
                    "operand of unary + must be of a numeric type"_err_en_US);
              }
            },
        },
        value->u);
  }
  return value;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Negate &x) {
  if (MaybeExpr operand{AnalyzeExpr(context, *x.v)}) {
    return Negation(context.GetContextualMessages(), std::move(*operand));
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::NOT &x) {
  if (MaybeExpr operand{AnalyzeExpr(context, *x.v)}) {
    return std::visit(
        common::visitors{
            [](Expr<SomeLogical> &&lx) -> MaybeExpr {
              return {AsGenericExpr(LogicalNegation(std::move(lx)))};
            },
            [&](auto &&) -> MaybeExpr {
              // TODO: accept INTEGER operand and maybe typeless
              // if not overridden
              context.Say("Operand of .NOT. must be LOGICAL"_err_en_US);
              return std::nullopt;
            },
        },
        std::move(operand->u));
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::PercentLoc &) {
  context.Say("TODO: %LOC unimplemented"_err_en_US);
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::DefinedUnary &) {
  context.Say("TODO: DefinedUnary unimplemented"_err_en_US);
  return std::nullopt;
}

// Binary (dyadic) operations

// TODO: check defined operators for illegal intrinsic operator cases
template<template<typename> class OPR, typename PARSED>
MaybeExpr BinaryOperationHelper(
    ExpressionAnalysisContext &context, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(context, *std::get<0>(x.t)),
          AnalyzeExpr(context, *std::get<1>(x.t)))}) {
    ConformabilityCheck(context.GetContextualMessages(), std::get<0>(*both),
        std::get<1>(*both));
    return NumericOperation<OPR>(context.GetContextualMessages(),
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both)),
        context.GetDefaultKind(TypeCategory::Real));
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Power &x) {
  return BinaryOperationHelper<Power>(context, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Multiply &x) {
  return BinaryOperationHelper<Multiply>(context, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Divide &x) {
  return BinaryOperationHelper<Divide>(context, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Add &x) {
  return BinaryOperationHelper<Add>(context, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Subtract &x) {
  return BinaryOperationHelper<Subtract>(context, x);
}

static MaybeExpr AnalyzeExpr(ExpressionAnalysisContext &context,
    const parser::Expr::ComplexConstructor &x) {
  auto re{AnalyzeExpr(context, *std::get<0>(x.t))};
  auto im{AnalyzeExpr(context, *std::get<1>(x.t))};
  if (re.has_value() && im.has_value()) {
    ConformabilityCheck(context.GetContextualMessages(), *re, *im);
  }
  return AsMaybeExpr(
      ConstructComplex(context.GetContextualMessages(), std::move(re),
          std::move(im), context.GetDefaultKind(TypeCategory::Real)));
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::Concat &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(context, *std::get<0>(x.t)),
          AnalyzeExpr(context, *std::get<1>(x.t)))}) {
    ConformabilityCheck(context.GetContextualMessages(), std::get<0>(*both),
        std::get<1>(*both));
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
                      context.Say(
                          "Operands of // must be the same kind of CHARACTER"_err_en_US);
                      return std::nullopt;
                    }
                  },
                  std::move(cx.u), std::move(cy.u));
            },
            [&](auto &&, auto &&) -> MaybeExpr {
              context.Say("Operands of // must be CHARACTER"_err_en_US);
              return std::nullopt;
            },
        },
        std::move(std::get<0>(*both).u), std::move(std::get<1>(*both).u));
  }
  return std::nullopt;
}

// TODO: check defined operators for illegal intrinsic operator cases
template<typename PARSED>
MaybeExpr RelationHelper(ExpressionAnalysisContext &context,
    RelationalOperator opr, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(context, *std::get<0>(x.t)),
          AnalyzeExpr(context, *std::get<1>(x.t)))}) {
    ConformabilityCheck(context.GetContextualMessages(), std::get<0>(*both),
        std::get<1>(*both));
    return AsMaybeExpr(Relate(context.GetContextualMessages(), opr,
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both))));
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::LT &x) {
  return RelationHelper(context, RelationalOperator::LT, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::LE &x) {
  return RelationHelper(context, RelationalOperator::LE, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::EQ &x) {
  return RelationHelper(context, RelationalOperator::EQ, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::NE &x) {
  return RelationHelper(context, RelationalOperator::NE, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::GE &x) {
  return RelationHelper(context, RelationalOperator::GE, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::GT &x) {
  return RelationHelper(context, RelationalOperator::GT, x);
}

// TODO: check defined operators for illegal intrinsic operator cases
template<typename PARSED>
MaybeExpr LogicalHelper(
    ExpressionAnalysisContext &context, LogicalOperator opr, const PARSED &x) {
  if (auto both{common::AllPresent(AnalyzeExpr(context, *std::get<0>(x.t)),
          AnalyzeExpr(context, *std::get<1>(x.t)))}) {
    return std::visit(
        common::visitors{
            [&](Expr<SomeLogical> &&lx, Expr<SomeLogical> &&ly) -> MaybeExpr {
              ConformabilityCheck(context.GetContextualMessages(), lx, ly);
              return {AsGenericExpr(
                  BinaryLogicalOperation(opr, std::move(lx), std::move(ly)))};
            },
            [&](auto &&, auto &&) -> MaybeExpr {
              // TODO: extension: INTEGER and typeless operands
              // ifort and PGI accept them if not overridden
              // need to define IAND, IOR, IEOR intrinsic representation
              context.Say(
                  "operands to LOGICAL operation must be LOGICAL"_err_en_US);
              return {};
            },
        },
        std::move(std::get<0>(*both).u), std::move(std::get<1>(*both).u));
  }
  return std::nullopt;
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::AND &x) {
  return LogicalHelper(context, LogicalOperator::And, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::OR &x) {
  return LogicalHelper(context, LogicalOperator::Or, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::EQV &x) {
  return LogicalHelper(context, LogicalOperator::Eqv, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::NEQV &x) {
  return LogicalHelper(context, LogicalOperator::Neqv, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::XOR &x) {
  return LogicalHelper(context, LogicalOperator::Neqv, x);
}

static MaybeExpr AnalyzeExpr(
    ExpressionAnalysisContext &context, const parser::Expr::DefinedBinary &) {
  context.Say("TODO: DefinedBinary unimplemented"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExpressionAnalysisContext::Analyze(const parser::Expr &expr) {
  if (const auto *typed{expr.typedExpr.get()}) {
    // Expression was already checked by AnalyzeExpressions() below.
    return std::make_optional<Expr<SomeType>>(typed->v);
  } else if (!expr.source.empty()) {
    // Analyze the expression in a specified source position context for better
    // error reporting.
    auto save{GetFoldingContext().messages.SetLocation(expr.source)};
    return AnalyzeExpr(*this, expr.u);
  } else {
    return AnalyzeExpr(*this, expr.u);
  }
}
MaybeExpr ExpressionAnalysisContext::Analyze(const parser::Variable &variable) {
  return AnalyzeExpr(*this, variable.u);
}

Expr<SubscriptInteger> ExpressionAnalysisContext::Analyze(TypeCategory category,
    const std::optional<parser::KindSelector> &selector) {
  int defaultKind{GetDefaultKind(category)};
  if (!selector.has_value()) {
    return Expr<SubscriptInteger>{defaultKind};
  }
  return std::visit(
      common::visitors{
          [&](const parser::ScalarIntConstantExpr &x)
              -> Expr<SubscriptInteger> {
            if (MaybeExpr kind{AnalyzeExpr(*this, x)}) {
              Expr<SomeType> folded{
                  Fold(GetFoldingContext(), std::move(*kind))};
              if (std::optional<std::int64_t> code{ToInt64(folded)}) {
                if (IsValidKindOfIntrinsicType(category, *code)) {
                  return Expr<SubscriptInteger>{*code};
                }
                SayAt(x, "%s(KIND=%jd) is not a supported type"_err_en_US,
                    parser::ToUpperCaseLetters(EnumToString(category)).data(),
                    *code);
              } else if (auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(folded)}) {
                return ConvertToType<SubscriptInteger>(std::move(*intExpr));
              }
            }
            return Expr<SubscriptInteger>{defaultKind};
          },
          [&](const parser::KindSelector::StarSize &x)
              -> Expr<SubscriptInteger> {
            std::intmax_t size = x.v;
            if (category == TypeCategory::Complex) {
              // COMPLEX*16 == COMPLEX(KIND=8)
              if ((size % 2) == 0 &&
                  evaluate::IsValidKindOfIntrinsicType(category, size / 2)) {
                size /= 2;
              } else {
                Say("COMPLEX*%jd is not a supported type"_err_en_US, size);
                size = defaultKind;
              }
            } else if (!evaluate::IsValidKindOfIntrinsicType(category, size)) {
              Say("%s*%jd is not a supported type"_err_en_US,
                  parser::ToUpperCaseLetters(EnumToString(category)).data(),
                  size);
              size = defaultKind;
            }
            return Expr<SubscriptInteger>{size};
          },
      },
      selector->u);
}

int ExpressionAnalysisContext::GetDefaultKind(common::TypeCategory category) {
  return context_.defaultKinds().GetDefaultKind(category);
}

DynamicType ExpressionAnalysisContext::GetDefaultKindOfType(
    common::TypeCategory category) {
  return {category, GetDefaultKind(category)};
}

bool ExpressionAnalysisContext::AddAcImpliedDo(
    parser::CharBlock name, int kind) {
  return acImpliedDos_.insert(std::make_pair(name, kind)).second;
}

void ExpressionAnalysisContext::RemoveAcImpliedDo(parser::CharBlock name) {
  auto iter{acImpliedDos_.find(name)};
  if (iter != acImpliedDos_.end()) {
    acImpliedDos_.erase(iter);
  }
}

std::optional<int> ExpressionAnalysisContext::IsAcImpliedDo(
    parser::CharBlock name) const {
  auto iter{acImpliedDos_.find(name)};
  if (iter != acImpliedDos_.cend()) {
    return {iter->second};
  } else {
    return std::nullopt;
  }
}
}

namespace Fortran::semantics {

namespace {
class Visitor {
public:
  Visitor(SemanticsContext &context) : context_{context} {}

  template<typename A> bool Pre(const A &) { return true /* visit children */; }
  template<typename A> void Post(const A &) {}

  bool Pre(const parser::Expr &expr) {
    if (expr.typedExpr.get() == nullptr) {
      if (MaybeExpr checked{AnalyzeExpr(context_, expr)}) {
#if PMKDEBUG
//      checked->AsFortran(std::cout << "checked expression: ") << '\n';
#endif
        expr.typedExpr.reset(
            new evaluate::GenericExprWrapper{std::move(*checked)});
      } else {
#if PMKDEBUG
        std::cout << "TODO: expression analysis failed for this expression: ";
        DumpTree(std::cout, expr);
#endif
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
};
}

void AnalyzeExpressions(parser::Program &program, SemanticsContext &context) {
  Visitor visitor{context};
  parser::Walk(program, visitor);
}

evaluate::Expr<evaluate::SubscriptInteger> AnalyzeKindSelector(
    SemanticsContext &context, parser::CharBlock source,
    common::TypeCategory category,
    const std::optional<parser::KindSelector> &selector) {
  evaluate::ExpressionAnalysisContext exprContext{context};
  auto save{exprContext.GetContextualMessages().SetLocation(source)};
  return exprContext.Analyze(category, selector);
}
}
