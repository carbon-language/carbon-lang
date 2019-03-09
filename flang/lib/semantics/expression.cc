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
#include "tools.h"
#include "../common/idioms.h"
#include "../evaluate/common.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/characters.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <functional>
#include <optional>
#include <set>

// TODO pmk remove when scaffolding is obsolete
#undef PMKDEBUG  // #define PMKDEBUG 1
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
          [](NullPointer &&) -> std::optional<DataRef> { return std::nullopt; },
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

struct DynamicTypeWithLength : public DynamicType {
  std::optional<Expr<SubscriptInteger>> LEN() const;
  std::optional<Expr<SubscriptInteger>> length;
};

std::optional<Expr<SubscriptInteger>> DynamicTypeWithLength::LEN() const {
  if (length.has_value()) {
    return length;
  }
  if (charLength != nullptr) {
    if (const auto &len{charLength->GetExplicit()}) {
      return ConvertToType<SubscriptInteger>(common::Clone(*len));
    }
  }
  return std::nullopt;
}

static std::optional<DynamicTypeWithLength> AnalyzeTypeSpec(
    const std::optional<parser::TypeSpec> &spec) {
  if (spec.has_value()) {
    if (const semantics::DeclTypeSpec * typeSpec{spec->declTypeSpec}) {
      // Name resolution sets TypeSpec::declTypeSpec only when it's valid
      // (viz., an intrinsic type with valid known kind or a non-polymorphic
      // & non-ABSTRACT derived type).
      if (const semantics::IntrinsicTypeSpec *
          intrinsic{typeSpec->AsIntrinsic()}) {
        TypeCategory category{intrinsic->category()};
        if (auto optKind{ToInt64(intrinsic->kind())}) {
          int kind{static_cast<int>(*optKind)};
          if (category == TypeCategory::Character) {
            const semantics::CharacterTypeSpec &cts{
                typeSpec->characterTypeSpec()};
            const semantics::ParamValue &len{cts.length()};
            // N.B. CHARACTER(LEN=*) is allowed in type-specs in ALLOCATE() &
            // type guards, but not in array constructors.
            return DynamicTypeWithLength{DynamicType{kind, len}};
          } else {
            return DynamicTypeWithLength{DynamicType{category, kind}};
          }
        }
      } else if (const semantics::DerivedTypeSpec *
          derived{typeSpec->AsDerived()}) {
        return DynamicTypeWithLength{DynamicType{*derived}};
      }
    }
  }
  return std::nullopt;
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
static MaybeExpr Designate(DataRef &&ref) {
  if (std::optional<DynamicType> dyType{
          GetSymbolType(ref.GetLastSymbol().GetUltimate())}) {
    return TypedWrapper<Designator, DataRef>(
        std::move(*dyType), std::move(ref));
  }
  // TODO: graceful errors on CLASS(*) and TYPE(*) misusage
  return std::nullopt;
}

// Some subscript semantic checks must be deferred until all of the
// subscripts are in hand.
MaybeExpr ExpressionAnalyzer::CompleteSubscripts(ArrayRef &&ref) {
  const Symbol &symbol{ref.GetLastSymbol().GetUltimate()};
  int symbolRank{symbol.Rank()};
  int subscripts = ref.size();
  if (subscripts == 0) {
    // A -> A(:,:)
    for (; subscripts < symbolRank; ++subscripts) {
      ref.emplace_back(Triplet{});
    }
  }
  if (subscripts != symbolRank) {
    Say("Reference to rank-%d object '%s' has %d subscripts"_err_en_US,
        symbolRank, symbol.name().ToString().data(), subscripts);
  } else if (subscripts == 0) {
    // nothing to check
  } else if (Component * component{std::get_if<Component>(&ref.base())}) {
    int baseRank{component->base().Rank()};
    if (baseRank > 0) {
      int subscriptRank{0};
      for (const auto &expr : ref.subscript()) {
        subscriptRank += expr.Rank();
      }
      if (subscriptRank > 0) {
        Say("Subscripts of component '%s' of rank-%d derived type "
            "array have rank %d but must all be scalar"_err_en_US,
            symbol.name().ToString().data(), baseRank, subscriptRank);
      }
    }
  } else if (const auto *details{
                 symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    // C928 & C1002
    if (Triplet * last{std::get_if<Triplet>(&ref.subscript().back().u)}) {
      if (!last->upper().has_value() && details->IsAssumedSize()) {
        Say("Assumed-size array '%s' must have explicit final "
            "subscript upper bound value"_err_en_US,
            symbol.name().ToString().data());
      }
    }
  }
  return Designate(DataRef{std::move(ref)});
}

// Applies subscripts to a data reference.
MaybeExpr ExpressionAnalyzer::ApplySubscripts(
    DataRef &&dataRef, std::vector<Subscript> &&subscripts) {
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) {
            return CompleteSubscripts(ArrayRef{*symbol, std::move(subscripts)});
          },
          [&](Component &&c) {
            return CompleteSubscripts(
                ArrayRef{std::move(c), std::move(subscripts)});
          },
          [&](auto &&) -> MaybeExpr {
            CHECK(!"bad base for ArrayRef");
            return std::nullopt;
          },
      },
      std::move(dataRef.u));
}

// Top-level checks for data references.  Unsubscripted whole array references
// get expanded -- e.g., MATRIX becomes MATRIX(:,:).
MaybeExpr ExpressionAnalyzer::TopLevelChecks(DataRef &&dataRef) {
  bool addSubscripts{false};
  if (Component * component{std::get_if<Component>(&dataRef.u)}) {
    const Symbol &symbol{component->GetLastSymbol()};
    int componentRank{symbol.Rank()};
    if (componentRank > 0) {
      int baseRank{component->base().Rank()};
      if (baseRank > 0) {
        Say("reference to whole rank-%d component '%%%s' of "
            "rank-%d array of derived type is not allowed"_err_en_US,
            componentRank, symbol.name().ToString().data(), baseRank);
      } else {
        addSubscripts = true;
      }
    }
  } else if (const Symbol **symbol{std::get_if<const Symbol *>(&dataRef.u)}) {
    addSubscripts = (*symbol)->Rank() > 0;
  }
  if (addSubscripts) {
    if (MaybeExpr subscripted{
            ApplySubscripts(std::move(dataRef), std::vector<Subscript>{})}) {
      return subscripted;
    }
  }
  return Designate(std::move(dataRef));
}

// Parse tree correction after a substring S(j:k) was misparsed as an
// array section.  N.B. Fortran substrings have to have a range, not a
// single index.
static void FixMisparsedSubstring(const parser::Designator &d) {
  auto &mutate{const_cast<parser::Designator &>(d)};
  if (auto *dataRef{std::get_if<parser::DataRef>(&mutate.u)}) {
    if (auto *ae{std::get_if<common::Indirection<parser::ArrayElement>>(
            &dataRef->u)}) {
      parser::ArrayElement &arrElement{ae->value()};
      if (!arrElement.subscripts.empty()) {
        auto iter{arrElement.subscripts.begin()};
        if (auto *triplet{std::get_if<parser::SubscriptTriplet>(&iter->u)}) {
          if (!std::get<2>(triplet->t).has_value() /* no stride */ &&
              ++iter == arrElement.subscripts.end() /* one subscript */) {
            if (Symbol *
                symbol{std::visit(
                    common::visitors{
                        [](parser::Name &n) { return n.symbol; },
                        [](common::Indirection<parser::StructureComponent>
                                &sc) { return sc.value().component.symbol; },
                        [](auto &) -> Symbol * { return nullptr; },
                    },
                    arrElement.base.u)}) {
              const Symbol &ultimate{symbol->GetUltimate()};
              if (const semantics::DeclTypeSpec *type{ultimate.GetType()}) {
                if (!ultimate.IsObjectArray() &&
                    type->category() == semantics::DeclTypeSpec::Character) {
                  // The ambiguous S(j:k) was parsed as an array section
                  // reference, but it's now clear that it's a substring.
                  // Fix the parse tree in situ.
                  mutate.u = arrElement.ConvertToSubstring();
                }
              }
            }
          }
        }
      }
    }
  }
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Designator &d) {
  FixMisparsedSubstring(d);
  // These checks have to be deferred to these "top level" data-refs where
  // we can be sure that there are no following subscripts (yet).
  if (MaybeExpr result{Analyze(d.u)}) {
    if (std::optional<evaluate::DataRef> dataRef{
            evaluate::ExtractDataRef(std::move(result))}) {
      return TopLevelChecks(std::move(*dataRef));
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
int ExpressionAnalyzer::AnalyzeKindParam(
    const std::optional<parser::KindParam> &kindParam, int defaultKind,
    int kanjiKind /* = -1 */) {
  if (!kindParam.has_value()) {
    return defaultKind;
  }
  return std::visit(
      common::visitors{
          [](std::uint64_t k) { return static_cast<int>(k); },
          [&](const parser::Scalar<
              parser::Integer<parser::Constant<parser::Name>>> &n) {
            if (MaybeExpr ie{Analyze(n)}) {
              if (std::optional<std::int64_t> i64{ToInt64(*ie)}) {
                int iv = *i64;
                if (iv == *i64) {
                  return iv;
                }
              }
            }
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
MaybeExpr ExpressionAnalyzer::IntLiteralConstant(const PARSED &x) {
  int kind{AnalyzeKindParam(std::get<std::optional<parser::KindParam>>(x.t),
      GetDefaultKind(TypeCategory::Integer))};
  auto value{std::get<0>(x.t)};  // std::(u)int64_t
  if (!CheckIntrinsicKind(TypeCategory::Integer, kind)) {
    return std::nullopt;
  }
  return common::SearchTypes(
      TypeKindVisitor<TypeCategory::Integer, Constant, std::int64_t>{
          kind, static_cast<std::int64_t>(value)});
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::IntLiteralConstant &x) {
  return IntLiteralConstant(x);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::SignedIntLiteralConstant &x) {
  return IntLiteralConstant(x);
}

template<typename TYPE>
Constant<TYPE> ReadRealLiteral(
    parser::CharBlock source, FoldingContext &context) {
  const char *p{source.begin()};
  auto valWithFlags{Scalar<TYPE>::Read(p, context.rounding())};
  CHECK(p == source.end());
  RealFlagWarnings(context, valWithFlags.flags, "conversion of REAL literal");
  auto value{valWithFlags.value};
  if (context.flushSubnormalsToZero()) {
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
MaybeExpr ExpressionAnalyzer::Analyze(const parser::RealLiteralConstant &x) {
  // Use a local message context around the real literal for better
  // provenance on any messages.
  auto save{GetContextualMessages().SetLocation(x.real.source)};
  // If a kind parameter appears, it defines the kind of the literal and any
  // letter used in an exponent part (e.g., the 'E' in "6.02214E+23")
  // should agree.  In the absence of an explicit kind parameter, any exponent
  // letter determines the kind.  Otherwise, defaults apply.
  // TODO: warn on inexact conversions?
  auto &defaults{context().defaultKinds()};
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
  auto kind{AnalyzeKindParam(x.kind, defaultKind)};
  if (letterKind.has_value() && kind != *letterKind) {
    Say("explicit kind parameter on real constant disagrees with "
        "exponent letter"_en_US);
  }
  auto result{common::SearchTypes(
      RealTypeVisitor{kind, x.real.source, GetFoldingContext()})};
  if (!result.has_value()) {
    Say("unsupported REAL(KIND=%d)"_err_en_US, kind);
  }
  return AsMaybeExpr(std::move(result));
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::SignedRealLiteralConstant &x) {
  if (auto result{Analyze(std::get<parser::RealLiteralConstant>(x.t))}) {
    auto &realExpr{std::get<Expr<SomeReal>>(result->u)};
    if (auto sign{std::get<std::optional<parser::Sign>>(x.t)}) {
      if (sign == parser::Sign::Negative) {
        return {AsGenericExpr(-std::move(realExpr))};
      }
    }
    return result;
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ComplexPart &x) {
  return Analyze(x.u);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ComplexLiteralConstant &z) {
  return AsMaybeExpr(
      ConstructComplex(GetContextualMessages(), Analyze(std::get<0>(z.t)),
          Analyze(std::get<1>(z.t)), GetDefaultKind(TypeCategory::Real)));
}

// CHARACTER literal processing.
MaybeExpr ExpressionAnalyzer::AnalyzeString(std::string &&string, int kind) {
  if (!CheckIntrinsicKind(TypeCategory::Character, kind)) {
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

MaybeExpr ExpressionAnalyzer::Analyze(const parser::CharLiteralConstant &x) {
  int kind{
      AnalyzeKindParam(std::get<std::optional<parser::KindParam>>(x.t), 1)};
  auto value{std::get<std::string>(x.t)};
  return AnalyzeString(std::move(value), kind);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::HollerithLiteralConstant &x) {
  int kind{GetDefaultKind(TypeCategory::Character)};
  auto value{x.v};
  return AnalyzeString(std::move(value), kind);
}

// .TRUE. and .FALSE. of various kinds
MaybeExpr ExpressionAnalyzer::Analyze(const parser::LogicalLiteralConstant &x) {
  auto kind{AnalyzeKindParam(std::get<std::optional<parser::KindParam>>(x.t),
      GetDefaultKind(TypeCategory::Logical))};
  bool value{std::get<bool>(x.t)};
  auto result{common::SearchTypes(
      TypeKindVisitor<TypeCategory::Logical, Constant, bool>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    Say("unsupported LOGICAL(KIND=%d)"_err_en_US, kind);
  }
  return result;
}

// BOZ typeless literals
MaybeExpr ExpressionAnalyzer::Analyze(const parser::BOZLiteralConstant &x) {
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
MaybeExpr ExpressionAnalyzer::Analyze(const parser::Name &n) {
  if (std::optional<int> kind{IsAcImpliedDo(n.source)}) {
    return AsMaybeExpr(ConvertToKind<TypeCategory::Integer>(
        *kind, AsExpr(ImpliedDoIndex{n.source})));
  } else if (n.symbol != nullptr) {
    const Symbol &ultimate{n.symbol->GetUltimate()};
    if (ultimate.attrs().test(semantics::Attr::PARAMETER)) {
      if (auto *details{ultimate.detailsIf<semantics::ObjectEntityDetails>()}) {
        if (auto &init{details->init()}) {
          return init;
        }
      }
      // TODO: enumerators, do they have the PARAMETER attribute?
    } else if (ultimate.detailsIf<semantics::TypeParamDetails>()) {
      // A bare reference to a derived type parameter (within a parameterized
      // derived type definition)
      return AsMaybeExpr(MakeTypeParamInquiry(&ultimate));
    } else if (MaybeExpr result{Designate(DataRef{*n.symbol})}) {
      return result;
    } else {
      Say(n.source, "not of a supported type and kind"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::NamedConstant &n) {
  if (MaybeExpr value{Analyze(n.v)}) {
    Expr<SomeType> folded{Fold(GetFoldingContext(), std::move(*value))};
    if (IsConstantExpr(folded)) {
      return {folded};
    }
    Say(n.v.source, "must be a constant"_err_en_US);
  }
  return std::nullopt;
}

// Substring references
std::optional<Expr<SubscriptInteger>> ExpressionAnalyzer::GetSubstringBound(
    const std::optional<parser::ScalarIntExpr> &bound) {
  if (bound.has_value()) {
    if (MaybeExpr expr{Analyze(*bound)}) {
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

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Substring &ss) {
  if (MaybeExpr baseExpr{Analyze(std::get<parser::DataRef>(ss.t))}) {
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
                  Substring{std::move(checked.value()), std::move(first),
                      std::move(last)});
            }
          }
          Say("substring may apply only to CHARACTER"_err_en_US);
        }
      }
    }
  }
  return std::nullopt;
}

// CHARACTER literal substrings
MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::CharLiteralConstantSubstring &x) {
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
std::optional<Expr<SubscriptInteger>> ExpressionAnalyzer::AsSubscript(
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

std::optional<Expr<SubscriptInteger>> ExpressionAnalyzer::TripletPart(
    const std::optional<parser::Subscript> &s) {
  if (s.has_value()) {
    return AsSubscript(Analyze(*s));
  }
  return std::nullopt;
}

std::optional<Subscript> ExpressionAnalyzer::AnalyzeSectionSubscript(
    const parser::SectionSubscript &ss) {
  return std::visit(
      common::visitors{
          [&](const parser::SubscriptTriplet &t) {
            return std::make_optional(Subscript{Triplet{
                TripletPart(std::get<0>(t.t)), TripletPart(std::get<1>(t.t)),
                TripletPart(std::get<2>(t.t))}});
          },
          [&](const auto &s) -> std::optional<Subscript> {
            if (auto subscriptExpr{AsSubscript(Analyze(s))}) {
              return {Subscript{std::move(*subscriptExpr)}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

std::vector<Subscript> ExpressionAnalyzer::AnalyzeSectionSubscripts(
    const std::list<parser::SectionSubscript> &sss) {
  std::vector<Subscript> subscripts;
  for (const auto &s : sss) {
    if (auto subscript{AnalyzeSectionSubscript(s)}) {
      subscripts.emplace_back(std::move(*subscript));
    }
  }
  return subscripts;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ArrayElement &ae) {
  std::vector<Subscript> subscripts{AnalyzeSectionSubscripts(ae.subscripts)};
  if (MaybeExpr baseExpr{Analyze(ae.base)}) {
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

// Type parameter inquiries apply to data references, but don't depend
// on any trailing (co)subscripts.
static SymbolOrComponent IgnoreAnySubscripts(
    Designator<SomeDerived> &&designator) {
  return std::visit(
      common::visitors{
          [](const Symbol *symbol) { return SymbolOrComponent{symbol}; },
          [](Component &&component) { return SymbolOrComponent{component}; },
          [](ArrayRef &&arrayRef) { return std::move(arrayRef.base()); },
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
MaybeExpr ExpressionAnalyzer::Analyze(const parser::StructureComponent &sc) {
  const auto &name{sc.component.source};
  if (MaybeExpr base{Analyze(sc.base)}) {
    Symbol *sym{sc.component.symbol};
    if (sym == nullptr) {
      Say(sc.component.source,
          "component name was not resolved to a symbol"_err_en_US);
    } else if (auto *dtExpr{UnwrapExpr<Expr<SomeDerived>>(*base)}) {
      const semantics::DerivedTypeSpec *dtSpec{nullptr};
      if (std::optional<DynamicType> dtDyTy{dtExpr->GetType()}) {
        dtSpec = dtDyTy->derived;
      }
      if (sym->detailsIf<semantics::TypeParamDetails>()) {
        if (auto *designator{UnwrapExpr<Designator<SomeDerived>>(*dtExpr)}) {
          if (std::optional<DynamicType> dyType{GetSymbolType(*sym)}) {
            if (dyType->category == TypeCategory::Integer) {
              return AsMaybeExpr(
                  common::SearchTypes(TypeParamInquiryVisitor{dyType->kind,
                      IgnoreAnySubscripts(std::move(*designator)), *sym}));
            }
          }
          Say(name, "type parameter is not INTEGER"_err_en_US);
        } else {
          Say(name,
              "type parameter inquiry must be applied to "
              "a designator"_err_en_US);
        }
      } else if (dtSpec == nullptr || dtSpec->scope() == nullptr) {
        Say(name,
            "TODO: base of component reference lacks a derived type"_err_en_US);
      } else if (std::optional<DataRef> dataRef{
                     ExtractDataRef(std::move(*dtExpr))}) {
        if (auto component{
                CreateComponent(std::move(*dataRef), *sym, *dtSpec->scope())}) {
          return Designate(DataRef{std::move(*component)});
        } else {
          Say(name, "component is not in scope of derived TYPE(%s)"_err_en_US,
              dtSpec->typeSymbol().name().ToString().data());
        }
      } else {
        Say(name,
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
        func.type = GetDefaultKindOfType(TypeCategory::Integer);
        return TypedWrapper<FunctionRef, ProcedureRef>(*func.type,
            ProcedureRef{ProcedureDesignator{std::move(func)},
                ActualArguments{ActualArgument{std::move(*base)}}});
      } else {
        common::die("unexpected kind");
      }
    } else {
      Say(name, "derived type required before component reference"_err_en_US);
    }
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::CoindexedNamedObject &co) {
  Say("TODO: CoindexedNamedObject unimplemented"_err_en_US);
  return std::nullopt;
}

int ExpressionAnalyzer::IntegerTypeSpecKind(
    const parser::IntegerTypeSpec &spec) {
  Expr<SubscriptInteger> value{
      AnalyzeKindSelector(TypeCategory::Integer, spec.v)};
  if (auto kind{ToInt64(value)}) {
    return static_cast<int>(*kind);
  }
  SayAt(spec, "Constant INTEGER kind value required here"_err_en_US);
  return GetDefaultKind(TypeCategory::Integer);
}

// Array constructors

class ArrayConstructorContext : private ExpressionAnalyzer {
public:
  ArrayConstructorContext(
      ExpressionAnalyzer &c, std::optional<DynamicTypeWithLength> &t)
    : ExpressionAnalyzer{c}, type_{t} {}
  ArrayConstructorContext(ArrayConstructorContext &) = default;
  void Push(MaybeExpr &&);
  void Add(const parser::AcValue &);
  std::optional<DynamicTypeWithLength> &type() const { return type_; }
  const ArrayConstructorValues<SomeType> &values() { return values_; }

private:
  template<int KIND, typename A>
  std::optional<Expr<Type<TypeCategory::Integer, KIND>>> GetSpecificIntExpr(
      const A &x) {
    if (MaybeExpr y{Analyze(x)}) {
      Expr<SomeInteger> *intExpr{UnwrapExpr<Expr<SomeInteger>>(*y)};
      CHECK(intExpr != nullptr);
      return ConvertToType<Type<TypeCategory::Integer, KIND>>(
          std::move(*intExpr));
    }
    return std::nullopt;
  }

  std::optional<DynamicTypeWithLength> &type_;
  bool explicitType_{type_.has_value()};
  std::optional<std::int64_t> constantLength_;
  ArrayConstructorValues<SomeType> values_;
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
    if (!type_.has_value()) {
      // If there is no explicit type-spec in an array constructor, the type
      // of the array is the declared type of all of the elements, which must
      // be well-defined and all match.
      // TODO: Possible language extension: use the most general type of
      // the values as the type of a numeric constructed array, convert all
      // of the other values to that type.  Alternative: let the first value
      // determine the type, and convert the others to that type.
      // TODO pmk: better type compatibility checks for derived types
      CHECK(!explicitType_);
      type_ = std::move(xType);
      constantLength_ = ToInt64(type_->length);
      values_.Push(std::move(*x));
    } else if (!explicitType_) {
      if (static_cast<const DynamicType &>(*type_) ==
          static_cast<const DynamicType &>(xType)) {
        values_.Push(std::move(*x));
        if (auto thisLen{ToInt64(xType.LEN())}) {
          if (constantLength_.has_value()) {
            if (context().warnOnNonstandardUsage() &&
                *thisLen != *constantLength_) {
              Say("Character literal in array constructor without explicit "
                  "type has different length than earlier element"_en_US);
            }
            if (*thisLen > *constantLength_) {
              // Language extension: use the longest literal to determine the
              // length of the array constructor's character elements, not the
              // first, when there is no explicit type.
              *constantLength_ = *thisLen;
              type_->length = xType.LEN();
            }
          } else {
            constantLength_ = *thisLen;
            type_->length = xType.LEN();
          }
        }
      } else {
        Say("Values in array constructor must have the same declared type "
            "when no explicit type appears"_err_en_US);
      }
    } else {
      if (auto cast{ConvertToType(*type_, std::move(*x))}) {
        values_.Push(std::move(*cast));
      } else {
        Say("Value in array constructor could not be converted to the type "
            "of the array"_err_en_US);
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
                GetSpecificIntExpr<IntType::kind>(std::get<0>(triplet.t))};
            std::optional<Expr<IntType>> upper{
                GetSpecificIntExpr<IntType::kind>(std::get<1>(triplet.t))};
            std::optional<Expr<IntType>> stride{
                GetSpecificIntExpr<IntType::kind>(std::get<2>(triplet.t))};
            if (lower.has_value() && upper.has_value()) {
              if (!stride.has_value()) {
                stride = Expr<IntType>{1};
              }
              if (!type_.has_value()) {
                type_ = DynamicTypeWithLength{IntType::GetType()};
              }
              ArrayConstructorContext nested{*this};
              parser::CharBlock name;
              nested.Push(Expr<SomeType>{
                  Expr<SomeInteger>{Expr<IntType>{ImpliedDoIndex{name}}}});
              values_.Push(ImpliedDo<SomeType>{name, std::move(*lower),
                  std::move(*upper), std::move(*stride),
                  std::move(nested.values_)});
            }
          },
          [&](const common::Indirection<parser::Expr> &expr) {
            auto restorer{
                GetContextualMessages().SetLocation(expr.value().source)};
            if (MaybeExpr v{Analyze(expr.value())}) {
              Push(std::move(*v));
            }
          },
          [&](const common::Indirection<parser::AcImpliedDo> &impliedDo) {
            const auto &control{
                std::get<parser::AcImpliedDoControl>(impliedDo.value().t)};
            const auto &bounds{
                std::get<parser::LoopBounds<parser::ScalarIntExpr>>(control.t)};
            parser::CharBlock name{bounds.name.thing.thing.source};
            int kind{IntType::kind};
            if (auto &its{std::get<std::optional<parser::IntegerTypeSpec>>(
                    control.t)}) {
              kind = IntegerTypeSpecKind(*its);
            }
            bool inserted{AddAcImpliedDo(name, kind)};
            if (!inserted) {
              SayAt(name,
                  "Implied DO index is active in surrounding implied DO loop "
                  "and cannot have the same name"_err_en_US);
            }
            std::optional<Expr<IntType>> lower{
                GetSpecificIntExpr<IntType::kind>(bounds.lower)};
            std::optional<Expr<IntType>> upper{
                GetSpecificIntExpr<IntType::kind>(bounds.upper)};
            std::optional<Expr<IntType>> stride{
                GetSpecificIntExpr<IntType::kind>(bounds.step)};
            ArrayConstructorContext nested{*this};
            for (const auto &value :
                std::get<std::list<parser::AcValue>>(impliedDo.value().t)) {
              nested.Add(value);
            }
            if (lower.has_value() && upper.has_value()) {
              if (!stride.has_value()) {
                stride = Expr<IntType>{1};
              }
              values_.Push(ImpliedDo<SomeType>{name, std::move(*lower),
                  std::move(*upper), std::move(*stride),
                  std::move(nested.values_)});
            }
            if (inserted) {
              RemoveAcImpliedDo(name);
            }
          },
      },
      x.u);
}

// Inverts a collection of generic ArrayConstructorValues<SomeType> that
// all happen to have the same actual type T into one ArrayConstructor<T>.
template<typename T>
ArrayConstructorValues<T> MakeSpecific(
    ArrayConstructorValues<SomeType> &&from) {
  ArrayConstructorValues<T> to;
  for (ArrayConstructorValue<SomeType> &x : from.values()) {
    std::visit(
        common::visitors{
            [&](CopyableIndirection<Expr<SomeType>> &&expr) {
              auto *typed{UnwrapExpr<Expr<T>>(expr.value())};
              CHECK(typed != nullptr);
              to.Push(std::move(*typed));
            },
            [&](ImpliedDo<SomeType> &&impliedDo) {
              to.Push(ImpliedDo<T>{impliedDo.name(),
                  std::move(impliedDo.lower()), std::move(impliedDo.upper()),
                  std::move(impliedDo.stride()),
                  MakeSpecific<T>(std::move(impliedDo.values()))});
            },
        },
        std::move(x.u));
  }
  return to;
}

struct ArrayConstructorTypeVisitor {
  using Result = MaybeExpr;
  using Types = AllTypes;
  template<typename T> Result Test() {
    if (type.category == T::category) {
      if constexpr (T::category == TypeCategory::Derived) {
        CHECK(type.derived != nullptr);
        return AsMaybeExpr(ArrayConstructor<T>{
            *type.derived, MakeSpecific<T>(std::move(values))});
      } else if (type.kind == T::kind) {
        if constexpr (T::category == TypeCategory::Character) {
          return AsMaybeExpr(ArrayConstructor<T>{
              type.LEN().value(), MakeSpecific<T>(std::move(values))});
        } else {
          return AsMaybeExpr(
              ArrayConstructor<T>{MakeSpecific<T>(std::move(values))});
        }
      }
    }
    return std::nullopt;
  }
  DynamicTypeWithLength type;
  ArrayConstructorValues<SomeType> values;
};

MaybeExpr ExpressionAnalyzer::Analyze(const parser::ArrayConstructor &array) {
  const parser::AcSpec &acSpec{array.v};
  std::optional<DynamicTypeWithLength> type{AnalyzeTypeSpec(acSpec.type)};
  ArrayConstructorContext context{*this, type};
  for (const parser::AcValue &value : acSpec.values) {
    context.Add(value);
  }
  if (type.has_value()) {
    ArrayConstructorTypeVisitor visitor{
        std::move(*type), std::move(context.values())};
    return common::SearchTypes(std::move(visitor));
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::StructureConstructor &structure) {
  auto &parsedType{std::get<parser::DerivedTypeSpec>(structure.t)};
  parser::CharBlock typeName{std::get<parser::Name>(parsedType.t).source};
  if (parsedType.derivedTypeSpec == nullptr) {
    Say("INTERNAL: parser::StructureConstructor lacks type"_err_en_US);
    return std::nullopt;
  }
  const auto &spec{*parsedType.derivedTypeSpec};
  CHECK(spec.scope() != nullptr);
  const Symbol &typeSymbol{spec.typeSymbol()};

  if (typeSymbol.attrs().test(semantics::Attr::ABSTRACT)) {  // C796
    if (auto *msg{Say(typeName,
            "ABSTRACT derived type '%s' cannot be used in a "
            "structure constructor"_err_en_US,
            typeName.ToString().data())}) {
      msg->Attach(
          typeSymbol.name(), "Declaration of ABSTRACT derived type"_en_US);
    }
  }

  // This list holds all of the components in the derived type and its
  // parents.  The symbols for whole parent components appear after their
  // own components and before the components of the types that extend them.
  // E.g., TYPE :: A; REAL X; END TYPE
  //       TYPE, EXTENDS(A) :: B; REAL Y; END TYPE
  // produces the component list X, A, Y.
  // The order is important below because a structure constructor can
  // initialize X or A by name, but not both.
  const auto &details{typeSymbol.get<semantics::DerivedTypeDetails>()};
  std::list<const Symbol *> components{details.OrderComponents(*spec.scope())};
  auto nextAnonymous{components.begin()};

  std::set<parser::CharBlock> unavailable;
  bool anyKeyword{false};
  StructureConstructor result{spec};
  bool checkConflicts{true};

  for (const auto &component :
      std::get<std::list<parser::ComponentSpec>>(structure.t)) {
    const parser::Expr &expr{
        std::get<parser::ComponentDataSource>(component.t).v.value()};
    parser::CharBlock source{expr.source};
    const Symbol *symbol{nullptr};
    if (const auto &kw{std::get<std::optional<parser::Keyword>>(component.t)}) {
      source = kw->v.source;
      symbol = kw->v.symbol;
      anyKeyword = true;
    } else {
      if (anyKeyword) {  // C7100
        Say(source,
            "Value in structure constructor lacks a component name"_err_en_US);
        checkConflicts = false;  // stem cascade
      }
      while (nextAnonymous != components.end()) {
        symbol = *nextAnonymous++;
        if (!symbol->test(Symbol::Flag::ParentComp)) {
          break;
        }
      }
      if (symbol == nullptr) {
        Say(source, "Unexpected value in structure constructor"_err_en_US);
      }
    }
    if (symbol != nullptr) {
      if (symbol->has<semantics::TypeParamDetails>()) {
        Say(source,
            "Type parameter '%s' cannot be a component of this structure "
            "constructor"_err_en_US,
            symbol->name().ToString().data());
        continue;
      }
      if (checkConflicts) {
        auto componentIter{
            std::find(components.begin(), components.end(), symbol)};
        if (unavailable.find(symbol->name()) != unavailable.cend()) {
          // C797, C798
          Say(source,
              "Component '%s' conflicts with another component earlier in "
              "this structure constructor"_err_en_US,
              symbol->name().ToString().data());
        } else if (symbol->test(Symbol::Flag::ParentComp)) {
          // Make earlier components unavailable once a whole parent appears.
          for (auto it{components.begin()}; it != componentIter; ++it) {
            unavailable.insert((*it)->name());
          }
        } else {
          // Make whole parent components unavailable after any of their
          // constituents appear.
          for (auto it{componentIter}; it != components.end(); ++it) {
            if ((*it)->test(Symbol::Flag::ParentComp)) {
              unavailable.insert((*it)->name());
            }
          }
        }
      }
      unavailable.insert(symbol->name());
      if (MaybeExpr value{Analyze(expr)}) {
        bool isNULL{std::holds_alternative<NullPointer>(value->u)};
        if (symbol->has<semantics::ProcEntityDetails>()) {
          CHECK(symbol->attrs().test(semantics::Attr::POINTER));
          if (!isNULL) {
            // TODO C7104: check that procedure pointer components are
            // being initialized with compatible procedure designators
            Say(expr.source,
                "TODO: non-null procedure pointer component value not implemented yet"_err_en_US);
          }
        } else {
          CHECK(symbol->has<semantics::ObjectEntityDetails>());
          // C1594(4)
          if (!isNULL) {
            const auto &innermost{context().FindScope(expr.source)};
            if (const auto *pureFunc{
                    semantics::FindPureFunctionContaining(&innermost)}) {
              if (const Symbol *
                  pointer{semantics::FindPointerComponent(*symbol)}) {
                if (const Symbol *
                    object{semantics::FindExternallyVisibleObject(
                        *value, *pureFunc)}) {
                  if (auto *msg{Say(expr.source,
                          "Externally visible object '%s' must not be "
                          "associated with pointer component '%s' in a "
                          "PURE function"_err_en_US,
                          object->name().ToString().data(),
                          pointer->name().ToString().data())}) {
                    msg->Attach(object->name(), "Object declaration"_en_US)
                        .Attach(pointer->name(), "Pointer declaration"_en_US);
                  }
                }
              }
            }
          }
          if (symbol->attrs().test(semantics::Attr::POINTER)) {
            if (!isNULL) {
              // TODO C7104: check that object pointer components are
              // being initialized with compatible object designators
            }
          } else if (MaybeExpr converted{
                         ConvertToType(*symbol, std::move(*value))}) {
            result.Add(*symbol, std::move(*converted));
          } else {
            if (auto *msg{Say(expr.source,
                    "Structure constructor value is incompatible with component '%s'"_err_en_US,
                    symbol->name().ToString().data())}) {
              msg->Attach(symbol->name(), "Component declaration"_en_US);
            }
          }
        }
      }
    }
  }

  // Ensure that unmentioned component objects have default initializers.
  for (const Symbol *symbol : components) {
    if (!symbol->test(Symbol::Flag::ParentComp) &&
        unavailable.find(symbol->name()) == unavailable.cend() &&
        !symbol->attrs().test(semantics::Attr::ALLOCATABLE)) {
      if (const auto *details{
              symbol->detailsIf<semantics::ObjectEntityDetails>()}) {
        if (details->init().has_value()) {
          result.Add(*symbol, common::Clone(*details->init()));
        } else {  // C799
          if (auto *msg{Say(typeName,
                  "Structure constructor lacks a value for "
                  "component '%s'"_err_en_US,
                  symbol->name().ToString().data())}) {
            msg->Attach(symbol->name(), "Absent component"_en_US);
          }
        }
      }
    }
  }

  return AsMaybeExpr(Expr<SomeDerived>{std::move(result)});
}

auto ExpressionAnalyzer::Procedure(const parser::ProcedureDesignator &pd,
    ActualArguments &arguments) -> std::optional<CallAndArguments> {
  return std::visit(
      common::visitors{
          [&](const parser::Name &n) -> std::optional<CallAndArguments> {
            if (n.symbol == nullptr) {
              Say("TODO INTERNAL no symbol for procedure designator name '%s'"_err_en_US,
                  n.ToString().data());
              return std::nullopt;
            }
            const Symbol &ultimate{n.symbol->GetUltimate()};
            if (const auto *proc{
                    ultimate.detailsIf<semantics::ProcEntityDetails>()}) {
              if (proc->HasExplicitInterface()) {
                // TODO: check actual arguments vs. interface
              } else {
                CallCharacteristics cc{n.source};
                if (std::optional<SpecificCall> specificCall{
                        context().intrinsics().Probe(
                            cc, arguments, &GetContextualMessages())}) {
                  return {
                      CallAndArguments{ProcedureDesignator{std::move(
                                           specificCall->specificIntrinsic)},
                          std::move(specificCall->arguments)}};
                } else {
                  // TODO: if name is not INTRINSIC, call with implicit
                  // interface
                }
              }
              return {CallAndArguments{
                  ProcedureDesignator{*n.symbol}, std::move(arguments)}};
            } else {
              Say(n.source, "not a procedure"_err_en_US);
              return std::nullopt;
            }
          },
          [&](const parser::ProcComponentRef &pcr)
              -> std::optional<CallAndArguments> {
            if (MaybeExpr component{Analyze(pcr.v)}) {
              // TODO distinguish PCR from TBP
              // TODO optional PASS argument for TBP
              return std::nullopt;
            } else {
              return std::nullopt;
            }
          },
      },
      pd.u);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::FunctionReference &funcRef) {
  // TODO: C1002: Allow a whole assumed-size array to appear if the dummy
  // argument would accept it.  Handle by special-casing the context
  // ActualArg -> Variable -> Designator.
  // TODO: Actual arguments that are procedures and procedure pointers need to
  // be detected and represented (they're not expressions).
  // TODO: C1534: Don't allow a "restricted" specific intrinsic to be passed.
  ActualArguments arguments;
  for (const auto &arg :
      std::get<std::list<parser::ActualArgSpec>>(funcRef.v.t)) {
    MaybeExpr actualArgExpr;
    std::visit(
        common::visitors{
            [&](const common::Indirection<parser::Variable> &v) {
              actualArgExpr = Analyze(v.value());
            },
            [&](const common::Indirection<parser::Expr> &x) {
              actualArgExpr = Analyze(x.value());
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
          Fold(GetFoldingContext(), std::move(*actualArgExpr))));
      if (const auto &argKW{std::get<std::optional<parser::Keyword>>(arg.t)}) {
        arguments.back()->keyword = argKW->v.source;
      }
    } else {
      return std::nullopt;
    }
  }

  // TODO: map user generic to specific procedure
  if (std::optional<CallAndArguments> proc{Procedure(
          std::get<parser::ProcedureDesignator>(funcRef.v.t), arguments)}) {
    if (std::optional<DynamicType> dyType{
            proc->procedureDesignator.GetType()}) {
      return TypedWrapper<FunctionRef, ProcedureRef>(*dyType,
          ProcedureRef{std::move(proc->procedureDesignator),
              std::move(proc->arguments)});
    } else {
      if (const auto *intrinsic{
              std::get_if<SpecificIntrinsic>(&proc->procedureDesignator.u)}) {
        if (intrinsic->name == "null"s && proc->arguments.empty()) {
          return {Expr<SomeType>{NullPointer{}}};
        }
      }
    }
  }
  return std::nullopt;
}

// Unary operations

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Parentheses &x) {
  // TODO: C1003: A parenthesized function reference may not return a
  // procedure pointer.
  if (MaybeExpr operand{Analyze(x.v.value())}) {
    return std::visit(
        common::visitors{
            [&](BOZLiteralConstant &&boz) {
              return operand;  // ignore parentheses around typeless constants
            },
            [&](NullPointer &&boz) {
              return operand;  // ignore parentheses around NULL()
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

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::UnaryPlus &x) {
  MaybeExpr value{Analyze(x.v.value())};
  if (value.has_value()) {
    std::visit(
        common::visitors{
            [](const BOZLiteralConstant &) {},  // allow +Z'1', it's harmless
            [&](const NullPointer &) {
              Say("+NULL() is not allowed"_err_en_US);
            },
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

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Negate &x) {
  if (MaybeExpr operand{Analyze(x.v.value())}) {
    return Negation(GetContextualMessages(), std::move(*operand));
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::NOT &x) {
  if (MaybeExpr operand{Analyze(x.v.value())}) {
    return std::visit(
        common::visitors{
            [](Expr<SomeLogical> &&lx) -> MaybeExpr {
              return {AsGenericExpr(LogicalNegation(std::move(lx)))};
            },
            [&](auto &&) -> MaybeExpr {
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

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::PercentLoc &) {
  Say("TODO: %LOC unimplemented"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::DefinedUnary &) {
  Say("TODO: DefinedUnary unimplemented"_err_en_US);
  return std::nullopt;
}

// Binary (dyadic) operations

// TODO: check defined operators for illegal intrinsic operator cases
template<template<typename> class OPR, typename PARSED>
MaybeExpr BinaryOperationHelper(ExpressionAnalyzer &context, const PARSED &x) {
  if (auto both{common::AllPresent(context.Analyze(std::get<0>(x.t).value()),
          context.Analyze(std::get<1>(x.t).value()))}) {
    ConformabilityCheck(context.GetContextualMessages(), std::get<0>(*both),
        std::get<1>(*both));
    return NumericOperation<OPR>(context.GetContextualMessages(),
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both)),
        context.GetDefaultKind(TypeCategory::Real));
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Power &x) {
  return BinaryOperationHelper<Power>(*this, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Multiply &x) {
  return BinaryOperationHelper<Multiply>(*this, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Divide &x) {
  return BinaryOperationHelper<Divide>(*this, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Add &x) {
  return BinaryOperationHelper<Add>(*this, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Subtract &x) {
  return BinaryOperationHelper<Subtract>(*this, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(
    const parser::Expr::ComplexConstructor &x) {
  auto re{Analyze(std::get<0>(x.t).value())};
  auto im{Analyze(std::get<1>(x.t).value())};
  if (re.has_value() && im.has_value()) {
    ConformabilityCheck(GetContextualMessages(), *re, *im);
  }
  return AsMaybeExpr(ConstructComplex(GetContextualMessages(), std::move(re),
      std::move(im), GetDefaultKind(TypeCategory::Real)));
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::Concat &x) {
  if (auto both{common::AllPresent(Analyze(std::get<0>(x.t).value()),
          Analyze(std::get<1>(x.t).value()))}) {
    ConformabilityCheck(
        GetContextualMessages(), std::get<0>(*both), std::get<1>(*both));
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
    ExpressionAnalyzer &context, RelationalOperator opr, const PARSED &x) {
  if (auto both{common::AllPresent(context.Analyze(std::get<0>(x.t).value()),
          context.Analyze(std::get<1>(x.t).value()))}) {
    ConformabilityCheck(context.GetContextualMessages(), std::get<0>(*both),
        std::get<1>(*both));
    return AsMaybeExpr(Relate(context.GetContextualMessages(), opr,
        std::move(std::get<0>(*both)), std::move(std::get<1>(*both))));
  }
  return std::nullopt;
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::LT &x) {
  return RelationHelper(*this, RelationalOperator::LT, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::LE &x) {
  return RelationHelper(*this, RelationalOperator::LE, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::EQ &x) {
  return RelationHelper(*this, RelationalOperator::EQ, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::NE &x) {
  return RelationHelper(*this, RelationalOperator::NE, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::GE &x) {
  return RelationHelper(*this, RelationalOperator::GE, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::GT &x) {
  return RelationHelper(*this, RelationalOperator::GT, x);
}

// TODO: check defined operators for illegal intrinsic operator cases
template<typename PARSED>
MaybeExpr LogicalHelper(
    ExpressionAnalyzer &context, LogicalOperator opr, const PARSED &x) {
  if (auto both{common::AllPresent(context.Analyze(std::get<0>(x.t).value()),
          context.Analyze(std::get<1>(x.t).value()))}) {
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

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::AND &x) {
  return LogicalHelper(*this, LogicalOperator::And, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::OR &x) {
  return LogicalHelper(*this, LogicalOperator::Or, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::EQV &x) {
  return LogicalHelper(*this, LogicalOperator::Eqv, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::NEQV &x) {
  return LogicalHelper(*this, LogicalOperator::Neqv, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::XOR &x) {
  return LogicalHelper(*this, LogicalOperator::Neqv, x);
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr::DefinedBinary &) {
  Say("TODO: DefinedBinary unimplemented"_err_en_US);
  return std::nullopt;
}

// Converts, if appropriate, an original misparse of ambiguous syntax like
// A(1) as a function reference into an array reference or a structure
// constructor.
template<typename... A>
void FixMisparsedFunctionReference(const std::variant<A...> &constU) {
  // The parse tree is updated in situ when resolving an ambiguous parse.
  using uType = std::decay_t<decltype(constU)>;
  auto &u{const_cast<uType &>(constU)};
  if (auto *func{
          std::get_if<common::Indirection<parser::FunctionReference>>(&u)}) {
    parser::FunctionReference &funcRef{func->value()};
    auto &proc{std::get<parser::ProcedureDesignator>(funcRef.v.t)};
    if (auto *name{std::get_if<parser::Name>(&proc.u)}) {
      if (name->symbol == nullptr) {
        return;
      }
      Symbol &symbol{name->symbol->GetUltimate()};
      if (symbol.has<semantics::ObjectEntityDetails>()) {
        if constexpr (common::HasMember<common::Indirection<parser::Designator>,
                          uType>) {
          u = common::Indirection{funcRef.ConvertToArrayElementRef()};
        } else {
          common::die("can't fix misparsed function as array reference");
        }
      } else if (symbol.has<semantics::DerivedTypeDetails>()) {
        if constexpr (common::HasMember<parser::StructureConstructor, uType>) {
          CHECK(symbol.scope() != nullptr);
          const semantics::DeclTypeSpec *type{
              symbol.scope()->FindInstantiatedDerivedType(
                  semantics::DerivedTypeSpec{symbol})};
          CHECK(type != nullptr);
          u = funcRef.ConvertToStructureConstructor(type->derivedTypeSpec());
        } else {
          common::die("can't fix misparsed function as structure constructor");
        }
      }
    }
  }
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Expr &expr) {
  if (expr.typedExpr.has_value()) {
    // Expression was already checked by ExprChecker
    return std::make_optional<Expr<SomeType>>(expr.typedExpr.value().v);
  } else {
    FixMisparsedFunctionReference(expr.u);
    if (!expr.source.empty()) {
      // Analyze the expression in a specified source position context for
      // better error reporting.
      auto save{GetFoldingContext().messages().SetLocation(expr.source)};
      return Analyze(expr.u);
    } else {
      return Analyze(expr.u);
    }
  }
}

MaybeExpr ExpressionAnalyzer::Analyze(const parser::Variable &variable) {
  FixMisparsedFunctionReference(variable.u);
  return Analyze(variable.u);
}

Expr<SubscriptInteger> ExpressionAnalyzer::AnalyzeKindSelector(
    TypeCategory category,
    const std::optional<parser::KindSelector> &selector) {
  int defaultKind{GetDefaultKind(category)};
  if (!selector.has_value()) {
    return Expr<SubscriptInteger>{defaultKind};
  }
  return std::visit(
      common::visitors{
          [&](const parser::ScalarIntConstantExpr &x)
              -> Expr<SubscriptInteger> {
            if (MaybeExpr kind{Analyze(x)}) {
              Expr<SomeType> folded{
                  Fold(GetFoldingContext(), std::move(*kind))};
              if (std::optional<std::int64_t> code{ToInt64(folded)}) {
                if (CheckIntrinsicKind(category, *code)) {
                  return Expr<SubscriptInteger>{*code};
                }
              } else if (auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(folded)}) {
                return ConvertToType<SubscriptInteger>(std::move(*intExpr));
              }
            }
            return Expr<SubscriptInteger>{defaultKind};
          },
          [&](const parser::KindSelector::StarSize &x)
              -> Expr<SubscriptInteger> {
            std::intmax_t size = x.v;
            if (!CheckIntrinsicSize(category, size)) {
              size = defaultKind;
            } else if (category == TypeCategory::Complex) {
              size /= 2;
            }
            return Expr<SubscriptInteger>{size};
          },
      },
      selector->u);
}

int ExpressionAnalyzer::GetDefaultKind(common::TypeCategory category) {
  return context_.defaultKinds().GetDefaultKind(category);
}

DynamicType ExpressionAnalyzer::GetDefaultKindOfType(
    common::TypeCategory category) {
  return {category, GetDefaultKind(category)};
}

bool ExpressionAnalyzer::CheckIntrinsicKind(
    TypeCategory category, std::int64_t kind) {
  if (IsValidKindOfIntrinsicType(category, kind)) {
    return true;
  } else {
    Say("%s(KIND=%jd) is not a supported type"_err_en_US,
        parser::ToUpperCaseLetters(EnumToString(category)).data(), kind);
    return false;
  }
}

bool ExpressionAnalyzer::CheckIntrinsicSize(
    TypeCategory category, std::int64_t size) {
  if (category == TypeCategory::Complex) {
    // COMPLEX*16 == COMPLEX(KIND=8)
    if (size % 2 == 0 && IsValidKindOfIntrinsicType(category, size / 2)) {
      return true;
    }
  } else if (IsValidKindOfIntrinsicType(category, size)) {
    return true;
  }
  Say("%s*%jd is not a supported type"_err_en_US,
      parser::ToUpperCaseLetters(EnumToString(category)).data(), size);
  return false;
}

bool ExpressionAnalyzer::AddAcImpliedDo(parser::CharBlock name, int kind) {
  return acImpliedDos_.insert(std::make_pair(name, kind)).second;
}

void ExpressionAnalyzer::RemoveAcImpliedDo(parser::CharBlock name) {
  auto iter{acImpliedDos_.find(name)};
  if (iter != acImpliedDos_.end()) {
    acImpliedDos_.erase(iter);
  }
}

std::optional<int> ExpressionAnalyzer::IsAcImpliedDo(
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

evaluate::Expr<evaluate::SubscriptInteger> AnalyzeKindSelector(
    SemanticsContext &context, common::TypeCategory category,
    const std::optional<parser::KindSelector> &selector) {
  evaluate::ExpressionAnalyzer analyzer{context};
  auto save{analyzer.GetContextualMessages().SetLocation(*context.location())};
  return analyzer.AnalyzeKindSelector(category, selector);
}

void ExprChecker::Enter(const parser::Expr &expr) {
  if (!expr.typedExpr.has_value()) {
    if (MaybeExpr checked{AnalyzeExpr(context_, expr)}) {
#if PMKDEBUG
//        checked->AsFortran(std::cout << "checked expression: ") << '\n';
#endif
      expr.typedExpr = new evaluate::GenericExprWrapper{std::move(*checked)};
    } else {
#if PMKDEBUG
      std::cout << "TODO: expression analysis failed for this expression: ";
      DumpTree(std::cout, expr);
#endif
    }
  }
}

void ExprChecker::Enter(const parser::Variable &var) {
#if PMKDEBUG
  if (MaybeExpr checked{AnalyzeExpr(context_, var)}) {
//    checked->AsFortran(std::cout << "checked variable: ") << '\n';
#else
  if (AnalyzeExpr(context_, var)) {
#endif
  } else {
#if PMKDEBUG
    std::cout << "TODO: expression analysis failed for this variable: ";
    DumpTree(std::cout, var);
#endif
  }
}
}
