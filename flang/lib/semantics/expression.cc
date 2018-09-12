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
#include "symbol.h"
#include "../common/idioms.h"
#include "../evaluate/common.h"
#include "../evaluate/tools.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <functional>
#include <iostream>  // TODO remove soon
#include <optional>

using namespace Fortran::parser::literals;

// Much of the code that implements semantic analysis of expressions is
// tightly coupled with their typed representations in lib/evaluate,
// and appears here in namespace Fortran::evaluate for convenience.
namespace Fortran::evaluate {

using common::TypeCategory;

using MaybeExpr = std::optional<Expr<SomeType>>;

// A utility subroutine to repackage optional expressions of various levels
// of type specificity as fully general MaybeExpr values.
template<typename A> MaybeExpr AsMaybeExpr(std::optional<A> &&x) {
  if (x.has_value()) {
    return {AsGenericExpr(AsCategoryExpr(AsExpr(std::move(*x))))};
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

template<TypeCategory CAT, int KIND>
MaybeExpr AsMaybeExpr(std::optional<Expr<Type<CAT, KIND>>> &&x) {
  if (x.has_value()) {
    return {AsGenericExpr(AsCategoryExpr(std::move(*x)))};
  }
  return std::nullopt;
}

// This local class wraps some state and a highly overloaded Analyze()
// member function that converts parse trees into (usually) generic
// expressions.
struct ExprAnalyzer {
  ExprAnalyzer(
      FoldingContext &ctx, const semantics::IntrinsicTypeDefaultKinds &dfts)
    : context{ctx}, defaults{dfts} {}

  MaybeExpr Analyze(const parser::Expr &);
  MaybeExpr Analyze(const parser::CharLiteralConstantSubstring &);
  MaybeExpr Analyze(const parser::LiteralConstant &);
  MaybeExpr Analyze(const parser::IntLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedIntLiteralConstant &);
  MaybeExpr Analyze(const parser::RealLiteralConstant &);
  MaybeExpr Analyze(const parser::SignedRealLiteralConstant &);
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
  MaybeExpr Analyze(const parser::TypeParamInquiry &);
  MaybeExpr Analyze(const parser::CoindexedNamedObject &);
  MaybeExpr Analyze(const parser::ComplexPart &);
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

  std::optional<Subscript> Analyze(const parser::SectionSubscript &);
  std::vector<Subscript> Analyze(const std::list<parser::SectionSubscript> &);

  std::optional<Expr<SubscriptInteger>> AsSubscript(MaybeExpr &&);
  std::optional<Expr<SubscriptInteger>> TripletPart(
      const std::optional<parser::Subscript> &);

  FoldingContext &context;
  const semantics::IntrinsicTypeDefaultKinds &defaults;
};

// This helper template function handles the Scalar<>, Integer<>, and
// Constant<> wrappers in the parse tree, as well as default behavior
// for unions.  (C++ doesn't allow template specialization in
// a class, so this helper template function must be outside ExprAnalyzer
// and reflect back into it.)
template<typename A> MaybeExpr AnalyzeHelper(ExprAnalyzer &ea, const A &x) {
  if constexpr (UnionTrait<A>) {
    return AnalyzeHelper(ea, x.u);
  } else {
    return ea.Analyze(x);
  }
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

// Implementations of ExprAnalyzer::Analyze follow for various parse tree
// node types.

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr &x) {
  return AnalyzeHelper(*this, x);
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

// Common handling of parser::IntLiteralConstant and SignedIntLiteralConstant
template<typename PARSED>
MaybeExpr IntLiteralConstant(ExprAnalyzer &ea, const PARSED &x) {
  int kind{ea.Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      ea.defaults.defaultIntegerKind)};
  auto value{std::get<0>(x.t)};  // std::(u)int64_t
  auto result{common::SearchDynamicTypes(
      TypeKindVisitor<TypeCategory::Integer, Constant, std::int64_t>{
          kind, static_cast<std::int64_t>(value)})};
  if (!result.has_value()) {
    ea.context.messages.Say("unsupported INTEGER(KIND=%d)"_err_en_US, kind);
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
    ctxMsgs.Say("unsupported REAL(KIND=%d)"_err_en_US, kind);
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

MaybeExpr ExprAnalyzer::Analyze(const parser::CharLiteralConstant &x) {
  int kind{Analyze(std::get<std::optional<parser::KindParam>>(x.t), 1)};
  auto value{std::get<std::string>(x.t)};
  auto result{common::SearchDynamicTypes(
      TypeKindVisitor<TypeCategory::Character, Constant, std::string>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    context.messages.Say("unsupported CHARACTER(KIND=%d)"_err_en_US, kind);
  }
  return result;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::LogicalLiteralConstant &x) {
  auto kind{Analyze(std::get<std::optional<parser::KindParam>>(x.t),
      defaults.defaultLogicalKind)};
  bool value{std::get<bool>(x.t)};
  auto result{common::SearchDynamicTypes(
      TypeKindVisitor<TypeCategory::Logical, Constant, bool>{
          kind, std::move(value)})};
  if (!result.has_value()) {
    context.messages.Say("unsupported LOGICAL(KIND=%d)"_err_en_US, kind);
  }
  return result;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::HollerithLiteralConstant &x) {
  return common::SearchDynamicTypes(
      TypeKindVisitor<TypeCategory::Character, Constant, std::string>{
          defaults.defaultCharacterKind, x.v});
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

template<typename TYPE, TypeCategory CATEGORY>
MaybeExpr DataRefIfType(
    const semantics::Symbol &symbol, int defaultKind, DataRef &&dataRef) {
  if (auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (details->type().has_value()) {
      if (details->type()->category() ==
          semantics::DeclTypeSpec::Category::Intrinsic) {
        std::uint64_t kindParam{
            details->type()->intrinsicTypeSpec().kind().value().value()};
        int kind = static_cast<int>(kindParam);
        if (static_cast<std::uint64_t>(kind) == kindParam) {
          // TODO: Inspection of semantics::IntrinsicTypeSpec requires the use
          // of forbidden RTTI via dynamic_cast<>.  See whether
          // semantics::IntrinsicTypeSpec can be augmented with query
          // interfaces instead.
          if (dynamic_cast<const TYPE *>(
                  &details->type()->intrinsicTypeSpec()) != nullptr) {
            if (kind == 0) {  // TODO: resolve default kinds in semantics
              kind = defaultKind;
            }
            if (MaybeExpr result{common::SearchDynamicTypes(
                    TypeKindVisitor<CATEGORY, DataReference, DataRef>{
                        kind, std::move(dataRef)})}) {
              return result;
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}

static MaybeExpr TypedDataRef(const semantics::Symbol &symbol,
    const semantics::IntrinsicTypeDefaultKinds &defaults, DataRef &&dataRef) {
  if (MaybeExpr result{
          DataRefIfType<semantics::IntegerTypeSpec, TypeCategory::Integer>(
              symbol, defaults.defaultIntegerKind, std::move(dataRef))}) {
    return result;
  }
  if (MaybeExpr result{
          DataRefIfType<semantics::RealTypeSpec, TypeCategory::Real>(
              symbol, defaults.defaultRealKind, std::move(dataRef))}) {
    return result;
  }
  if (MaybeExpr result{
          DataRefIfType<semantics::ComplexTypeSpec, TypeCategory::Complex>(
              symbol, defaults.defaultRealKind, std::move(dataRef))}) {
    return result;
  }
  if (MaybeExpr result{
          DataRefIfType<semantics::CharacterTypeSpec, TypeCategory::Character>(
              symbol, defaults.defaultCharacterKind, std::move(dataRef))}) {
    return result;
  }
  if (MaybeExpr result{
          DataRefIfType<semantics::LogicalTypeSpec, TypeCategory::Logical>(
              symbol, defaults.defaultLogicalKind, std::move(dataRef))}) {
    return result;
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Name &n) {
  if (n.symbol == nullptr) {
    // TODO: convert this to a CHECK later
    context.messages.Say(
        "TODO: name '%s' is not resolved to an object"_err_en_US,
        n.ToString().data());
  } else if (n.symbol->attrs().test(semantics::Attr::PARAMETER)) {
    context.messages.Say(
        "TODO: PARAMETER references not yet implemented"_err_en_US);
    // TODO: enumerators, do they have the PARAMETER attribute?
  } else {
    if (MaybeExpr result{
            TypedDataRef(*n.symbol, defaults, DataRef{*n.symbol})}) {
      return result;
    }
    context.messages.Say("'%s' is not of a supported type and kind"_err_en_US,
        n.ToString().data());
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::NamedConstant &n) {
  if (MaybeExpr value{Analyze(n.v)}) {
    if (std::optional<Constant<SomeType>> folded{value->Fold(context)}) {
      return {AsGenericExpr(std::move(*folded))};
    }
    context.messages.Say(
        "'%s' must be a constant"_err_en_US, n.v.ToString().data());
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Substring &ss) {
  context.messages.Say("TODO: Substring unimplemented\n"_err_en_US);
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> ExprAnalyzer::AsSubscript(
    MaybeExpr &&expr) {
  if (expr.has_value()) {
    if (auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
      if (auto *ssIntExpr{std::get_if<Expr<SubscriptInteger>>(&intExpr->u)}) {
        return {std::move(*ssIntExpr)};
      }
      return {Expr<SubscriptInteger>{
          Convert<SubscriptInteger, TypeCategory::Integer>{
              std::move(*intExpr)}}};
    } else {
      context.messages.Say("subscript expression is not INTEGER"_err_en_US);
    }
  }
  return std::nullopt;
}

std::optional<Expr<SubscriptInteger>> ExprAnalyzer::TripletPart(
    const std::optional<parser::Subscript> &s) {
  if (s.has_value()) {
    return AsSubscript(AnalyzeHelper(*this, *s));
  }
  return std::nullopt;
}

std::optional<Subscript> ExprAnalyzer::Analyze(
    const parser::SectionSubscript &ss) {
  return std::visit(
      common::visitors{[&](const parser::SubscriptTriplet &t) {
                         return std::make_optional(
                             Subscript{Triplet{TripletPart(std::get<0>(t.t)),
                                 TripletPart(std::get<1>(t.t)),
                                 TripletPart(std::get<2>(t.t))}});
                       },
          [&](const auto &s) -> std::optional<Subscript> {
            if (auto subscriptExpr{AsSubscript(AnalyzeHelper(*this, s))}) {
              return {Subscript{std::move(*subscriptExpr)}};
            } else {
              return std::nullopt;
            }
          }},
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

MaybeExpr ExprAnalyzer::Analyze(const parser::ArrayElement &ae) {
  std::vector<Subscript> subscripts{Analyze(ae.subscripts)};
  if (const parser::Name * name{std::get_if<parser::Name>(&ae.base.u)}) {
    if (name->symbol == nullptr) {
      // TODO: convert this to a CHECK later
      context.messages.Say(
          "TODO: name (%s) is not resolved to an object"_err_en_US,
          name->ToString().data());
    } else {
      ArrayRef arrayRef{*name->symbol, std::move(subscripts)};
      return TypedDataRef(
          *name->symbol, defaults, DataRef{std::move(arrayRef)});
    }
  } else if (const auto *component{
                 std::get_if<common::Indirection<parser::StructureComponent>>(
                     &ae.base.u)}) {
    // pmk continue development here
  } else {
    CHECK(!"parser::ArrayRef base DataRef is neither Name nor "
           "StructureComponent");
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::StructureComponent &sc) {
  context.messages.Say("TODO: StructureComponent unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::TypeParamInquiry &tpi) {
  context.messages.Say("TODO: TypeParamInquiry unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::CoindexedNamedObject &co) {
  context.messages.Say("TODO: CoindexedNamedObject unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::CharLiteralConstantSubstring &) {
  context.messages.Say(
      "TODO: CharLiteralConstantSubstring unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::ArrayConstructor &) {
  context.messages.Say("TODO: ArrayConstructor unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::StructureConstructor &) {
  context.messages.Say("TODO: StructureConstructor unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::FunctionReference &) {
  context.messages.Say("TODO: FunctionReference unimplemented\n"_err_en_US);
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
  MaybeExpr value{AnalyzeHelper(*this, *x.v)};
  if (value.has_value()) {
    std::visit(
        common::visitors{
            [](const BOZLiteralConstant &) {},  // allow +Z'1', it's harmless
            [&](const auto &catExpr) {
              TypeCategory cat{ResultType<decltype(catExpr)>::category};
              if (cat != TypeCategory::Integer && cat != TypeCategory::Real &&
                  cat != TypeCategory::Complex) {
                context.messages.Say(
                    "operand of unary + must be of a numeric type"_err_en_US);
              }
            }},
        value->u);
  }
  return value;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Negate &x) {
  if (MaybeExpr operand{AnalyzeHelper(*this, *x.v)}) {
    return Negation(context.messages, std::move(operand->u));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::NOT &x) {
  if (MaybeExpr operand{AnalyzeHelper(*this, *x.v)}) {
    return std::visit(common::visitors{[](Expr<SomeLogical> &&lx) -> MaybeExpr {
                                         return {AsGenericExpr(
                                             LogicalNegation(std::move(lx)))};
                                       },
                          [=](auto &&) -> MaybeExpr {
                            // TODO: accept INTEGER operand if not overridden
                            context.messages.Say(
                                "Operand of .NOT. must be LOGICAL"_err_en_US);
                            return std::nullopt;
                          }},
        std::move(operand->u));
  }
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::PercentLoc &) {
  context.messages.Say("TODO: %LOC unimplemented\n"_err_en_US);
  return std::nullopt;
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::DefinedUnary &) {
  context.messages.Say("TODO: DefinedUnary unimplemented\n"_err_en_US);
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
  return AsMaybeExpr(ConstructComplex(context.messages,
      AnalyzeHelper(*this, *std::get<0>(x.t)),
      AnalyzeHelper(*this, *std::get<1>(x.t))));
}

MaybeExpr ExprAnalyzer::Analyze(const parser::Expr::Concat &x) {
  if (auto both{common::AllPresent(AnalyzeHelper(*this, *std::get<0>(x.t)),
          AnalyzeHelper(*this, *std::get<1>(x.t)))}) {
    return std::visit(
        common::visitors{
            [&](Expr<SomeCharacter> &&cx, Expr<SomeCharacter> &&cy) {
              return std::visit(
                  [&](auto &&cxk, auto &&cyk) -> MaybeExpr {
                    using Ty = ResultType<decltype(cxk)>;
                    if constexpr (std::is_same_v<Ty,
                                      ResultType<decltype(cyk)>>) {
                      return {AsGenericExpr(AsCategoryExpr(AsExpr(
                          Concat<Ty::kind>{std::move(cxk), std::move(cyk)})))};
                    } else {
                      context.messages.Say(
                          "Operands of // must be the same kind of CHARACTER"_err_en_US);
                      return std::nullopt;
                    }
                  },
                  std::move(cx.u), std::move(cy.u));
            },
            [&](auto &&, auto &&) -> MaybeExpr {
              context.messages.Say(
                  "Operands of // must be CHARACTER"_err_en_US);
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
  if (auto both{common::AllPresent(AnalyzeHelper(ea, *std::get<0>(x.t)),
          AnalyzeHelper(ea, *std::get<1>(x.t)))}) {
    return AsMaybeExpr(Relate(ea.context.messages, opr,
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
  if (auto both{common::AllPresent(AnalyzeHelper(ea, *std::get<0>(x.t)),
          AnalyzeHelper(ea, *std::get<1>(x.t)))}) {
    return std::visit(
        common::visitors{
            [=](Expr<SomeLogical> &&lx, Expr<SomeLogical> &&ly) -> MaybeExpr {
              return {AsGenericExpr(
                  BinaryLogicalOperation(opr, std::move(lx), std::move(ly)))};
            },
            [&](auto &&, auto &&) -> MaybeExpr {
              // TODO: extension: INTEGER and typeless operands
              // ifort and PGI accept them if not overridden
              ea.context.messages.Say(
                  "operands to LOGICAL operation must be LOGICAL"_err_en_US);
              return {};
            }},
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
  context.messages.Say("TODO: DefinedBinary unimplemented\n"_err_en_US);
  return std::nullopt;
}

}  // namespace Fortran::evaluate

namespace Fortran::semantics {

evaluate::MaybeExpr AnalyzeExpr(evaluate::FoldingContext &context,
    const IntrinsicTypeDefaultKinds &defaults, const parser::Expr &expr) {
  return evaluate::ExprAnalyzer{context, defaults}.Analyze(expr);
}

class Mutator {
public:
  Mutator(evaluate::FoldingContext &context,
      const IntrinsicTypeDefaultKinds &defaults)
    : context_{context}, defaults_{defaults} {}

  template<typename A> bool Pre(A &) { return true /* visit children */; }
  template<typename A> void Post(A &) {}

  bool Pre(parser::Expr &expr) {
    if (expr.typedExpr.get() == nullptr) {
      if (MaybeExpr checked{AnalyzeExpr(context_, defaults_, expr)}) {
        checked->Dump(std::cout << "checked expression: ") << '\n';
        expr.typedExpr.reset(
            new evaluate::GenericExprWrapper{std::move(*checked)});
      } else {
        std::cout << "expression analysis failed for this expression: ";
        DumpTree(std::cout, expr);
      }
    }
    return false;
  }

private:
  evaluate::FoldingContext &context_;
  const IntrinsicTypeDefaultKinds &defaults_;
};

void AnalyzeExpressions(parser::Program &program,
    evaluate::FoldingContext &context,
    const IntrinsicTypeDefaultKinds &defaults) {
  Mutator mutator{context, defaults};
  parser::Walk(program, mutator);
}

}  // namespace Fortran::semantics
