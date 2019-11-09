// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "characteristics.h"
#include "check-expression.h"
#include "fold.h"
#include "intrinsics.h"
#include "tools.h"
#include "type.h"
#include "../common/indirection.h"
#include "../parser/message.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include <initializer_list>
#include <ostream>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate::characteristics {

// Copy attributes from a symbol to dst based on the mapping in pairs.
template<typename A, typename B>
static void CopyAttrs(const semantics::Symbol &src, A &dst,
    const std::initializer_list<std::pair<semantics::Attr, B>> &pairs) {
  for (const auto &pair : pairs) {
    if (src.attrs().test(pair.first)) {
      dst.attrs.set(pair.second);
    }
  }
}

// Shapes of function results and dummy arguments have to have
// the same rank, the same deferred dimensions, and the same
// values for explicit dimensions when constant.
static bool ShapesAreCompatible(const Shape &x, const Shape &y) {
  if (x.size() != y.size()) {
    return false;
  }
  auto yIter{y.begin()};
  for (const auto &xDim : x) {
    const auto &yDim{*yIter++};
    if (xDim.has_value() != yDim.has_value()) {
      return false;
    }
    if (xDim) {
      auto xConst{ToInt64(*xDim)};
      auto yConst{ToInt64(*yDim)};
      if (xConst.has_value() != yConst.has_value() ||
          (xConst && *xConst != *yConst)) {
        return false;
      }
    }
  }
  return true;
}

bool TypeAndShape::operator==(const TypeAndShape &that) const {
  return type_ == that.type_ && ShapesAreCompatible(shape_, that.shape_) &&
      attrs_ == that.attrs_ && corank_ == that.corank_;
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::Symbol &symbol) {
  return std::visit(
      common::visitors{
          [&](const semantics::ObjectEntityDetails &object) {
            return Characterize(object);
          },
          [&](const semantics::ProcEntityDetails &proc) {
            const semantics::ProcInterface &interface{proc.interface()};
            if (interface.type()) {
              return Characterize(*interface.type());
            } else {
              return Characterize(*interface.symbol());
            }
          },
          [&](const semantics::UseDetails &use) {
            return Characterize(use.symbol());
          },
          [&](const semantics::HostAssocDetails &assoc) {
            return Characterize(assoc.symbol());
          },
          [](const semantics::AssocEntityDetails &assoc) {
            if (const semantics::Symbol *
                nested{UnwrapWholeSymbolDataRef(assoc.expr())}) {
              return Characterize(*nested);
            } else {
              return std::optional<TypeAndShape>{};
            }
          },
          [](const auto &) { return std::optional<TypeAndShape>{}; },
      },
      symbol.details());
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::ObjectEntityDetails &object) {
  if (auto type{DynamicType::From(object.type())}) {
    TypeAndShape result{std::move(*type)};
    result.AcquireShape(object);
    return result;
  } else {
    return std::nullopt;
  }
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const semantics::DeclTypeSpec &spec) {
  if (auto type{DynamicType::From(spec)}) {
    return TypeAndShape{std::move(*type)};
  } else {
    return std::nullopt;
  }
}

std::optional<TypeAndShape> TypeAndShape::Characterize(
    const Expr<SomeType> &expr, FoldingContext &context) {
  if (const auto *symbol{UnwrapWholeSymbolDataRef(expr)}) {
    if (const auto *object{
            symbol->detailsIf<semantics::ObjectEntityDetails>()}) {
      return Characterize(*object);
    }
  }
  if (auto type{expr.GetType()}) {
    if (auto shape{GetShape(context, expr)}) {
      TypeAndShape result{*type, std::move(*shape)};
      if (type->category() == TypeCategory::Character) {
        if (const auto *chExpr{UnwrapExpr<Expr<SomeCharacter>>(expr)}) {
          if (auto length{chExpr->LEN()}) {
            result.set_LEN(Expr<SomeInteger>{std::move(*length)});
          }
        }
      }
      return result;
    }
  }
  return std::nullopt;
}

bool TypeAndShape::IsCompatibleWith(parser::ContextualMessages &messages,
    const TypeAndShape &that, const char *thisIs, const char *thatIs,
    bool isElemental) const {
  const auto &len{that.LEN()};
  if (!type_.IsTypeCompatibleWith(that.type_)) {
    std::stringstream lenstr;
    if (len) {
      len->AsFortran(lenstr);
    }
    messages.Say(
        "%1$s type '%2$s' is not compatible with %3$s type '%4$s'"_err_en_US,
        thatIs, that.type_.AsFortran(lenstr.str()), thisIs, type_.AsFortran());
    return false;
  }
  return isElemental ||
      CheckConformance(messages, shape_, that.shape_, thisIs, thatIs);
}

void TypeAndShape::AcquireShape(const semantics::ObjectEntityDetails &object) {
  CHECK(shape_.empty() && !attrs_.test(Attr::AssumedRank));
  corank_ = object.coshape().Rank();
  if (object.IsAssumedRank()) {
    attrs_.set(Attr::AssumedRank);
    return;
  }
  if (object.IsAssumedShape()) {
    attrs_.set(Attr::AssumedShape);
  }
  if (object.IsAssumedSize()) {
    attrs_.set(Attr::AssumedSize);
  }
  if (object.IsDeferredShape()) {
    attrs_.set(Attr::DeferredShape);
  }
  if (object.IsCoarray()) {
    attrs_.set(Attr::Coarray);
  }
  for (const semantics::ShapeSpec &dim : object.shape()) {
    if (dim.ubound().GetExplicit()) {
      Expr<SubscriptInteger> extent{*dim.ubound().GetExplicit()};
      if (dim.lbound().GetExplicit()) {
        extent = std::move(extent) +
            common::Clone(*dim.lbound().GetExplicit()) -
            Expr<SubscriptInteger>{1};
      }
      shape_.emplace_back(std::move(extent));
    } else {
      shape_.push_back(std::nullopt);
    }
  }
}

void TypeAndShape::AcquireLEN() {
  if (type_.category() == TypeCategory::Character) {
    if (const auto *param{type_.charLength()}) {
      if (const auto &intExpr{param->GetExplicit()}) {
        LEN_ = *intExpr;
      }
    }
  }
}

std::ostream &TypeAndShape::Dump(std::ostream &o) const {
  std::stringstream LENstr;
  if (LEN_) {
    LEN_->AsFortran(LENstr);
  }
  o << type_.AsFortran(LENstr.str());
  attrs_.Dump(o, EnumToString);
  if (!shape_.empty()) {
    o << " dimension(";
    char sep{'('};
    for (const auto &expr : shape_) {
      o << sep;
      sep = ',';
      if (expr) {
        expr->AsFortran(o);
      } else {
        o << ':';
      }
    }
    o << ')';
  }
  return o;
}

bool DummyDataObject::operator==(const DummyDataObject &that) const {
  return type == that.type && attrs == that.attrs && intent == that.intent &&
      coshape == that.coshape;
}

static common::Intent GetIntent(const semantics::Attrs &attrs) {
  if (attrs.test(semantics::Attr::INTENT_IN)) {
    return common::Intent::In;
  } else if (attrs.test(semantics::Attr::INTENT_OUT)) {
    return common::Intent::Out;
  } else if (attrs.test(semantics::Attr::INTENT_INOUT)) {
    return common::Intent::InOut;
  } else {
    return common::Intent::Default;
  }
}

std::optional<DummyDataObject> DummyDataObject::Characterize(
    const semantics::Symbol &symbol) {
  if (const auto *obj{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (auto type{TypeAndShape::Characterize(*obj)}) {
      std::optional<DummyDataObject> result{std::move(*type)};
      using semantics::Attr;
      CopyAttrs<DummyDataObject, DummyDataObject::Attr>(symbol, *result,
          {
              {Attr::OPTIONAL, DummyDataObject::Attr::Optional},
              {Attr::ALLOCATABLE, DummyDataObject::Attr::Allocatable},
              {Attr::ASYNCHRONOUS, DummyDataObject::Attr::Asynchronous},
              {Attr::CONTIGUOUS, DummyDataObject::Attr::Contiguous},
              {Attr::VALUE, DummyDataObject::Attr::Value},
              {Attr::VOLATILE, DummyDataObject::Attr::Volatile},
              {Attr::POINTER, DummyDataObject::Attr::Pointer},
              {Attr::TARGET, DummyDataObject::Attr::Target},
          });
      result->intent = GetIntent(symbol.attrs());
      return result;
    }
  }
  return std::nullopt;
}

bool DummyDataObject::CanBePassedViaImplicitInterface() const {
  if ((attrs &
          Attrs{Attr::Allocatable, Attr::Asynchronous, Attr::Optional,
              Attr::Pointer, Attr::Target, Attr::Value, Attr::Volatile})
          .any()) {
    return false;  // 15.4.2.2(3)(a)
  } else if ((type.attrs() &
                 TypeAndShape::Attrs{TypeAndShape::Attr::AssumedShape,
                     TypeAndShape::Attr::AssumedRank,
                     TypeAndShape::Attr::Coarray})
                 .any()) {
    return false;  // 15.4.2.2(3)(b-d)
  } else if (type.type().IsPolymorphic()) {
    return false;  // 15.4.2.2(3)(f)
  } else if (type.type().category() == TypeCategory::Derived) {
    if (!type.type().GetDerivedTypeSpec().parameters().empty()) {
      return false;  // 15.4.2.2(3)(e)
    }
  }
  return true;
}

std::ostream &DummyDataObject::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (intent != common::Intent::Default) {
    o << "INTENT(" << common::EnumToString(intent) << ')';
  }
  type.Dump(o);
  if (!coshape.empty()) {
    char sep{'['};
    for (const auto &expr : coshape) {
      expr.AsFortran(o << sep);
      sep = ',';
    }
  }
  return o;
}

DummyProcedure::DummyProcedure(Procedure &&p)
  : procedure{new Procedure{std::move(p)}} {}

bool DummyProcedure::operator==(const DummyProcedure &that) const {
  return attrs == that.attrs && intent == that.intent &&
      procedure.value() == that.procedure.value();
}

std::optional<DummyProcedure> DummyProcedure::Characterize(
    const semantics::Symbol &symbol, const IntrinsicProcTable &intrinsics) {
  if (auto procedure{Procedure::Characterize(symbol, intrinsics)}) {
    // Dummy procedures may not be elemental.  Elemental dummy procedure
    // interfaces are errors when the interface is not intrinsic, and that
    // error is caught elsewhere.  Elemental intrinsic interfaces are
    // made non-elemental.
    procedure->attrs.reset(Procedure::Attr::Elemental);
    DummyProcedure result{std::move(procedure.value())};
    CopyAttrs<DummyProcedure, DummyProcedure::Attr>(symbol, result,
        {
            {semantics::Attr::OPTIONAL, DummyProcedure::Attr::Optional},
            {semantics::Attr::POINTER, DummyProcedure::Attr::Pointer},
        });
    result.intent = GetIntent(symbol.attrs());
    return result;
  } else {
    return std::nullopt;
  }
}

std::ostream &DummyProcedure::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (intent != common::Intent::Default) {
    o << "INTENT(" << common::EnumToString(intent) << ')';
  }
  procedure.value().Dump(o);
  return o;
}

std::ostream &AlternateReturn::Dump(std::ostream &o) const { return o << '*'; }

DummyArgument::~DummyArgument() {}

bool DummyArgument::operator==(const DummyArgument &that) const {
  return u == that.u;
}

std::optional<DummyArgument> DummyArgument::Characterize(
    const semantics::Symbol &symbol, const IntrinsicProcTable &intrinsics) {
  auto name{symbol.name().ToString()};
  if (symbol.has<semantics::ObjectEntityDetails>()) {
    if (auto obj{DummyDataObject::Characterize(symbol)}) {
      return DummyArgument{std::move(name), std::move(obj.value())};
    }
  } else if (auto proc{DummyProcedure::Characterize(symbol, intrinsics)}) {
    return DummyArgument{std::move(name), std::move(proc.value())};
  }
  return std::nullopt;
}

std::optional<DummyArgument> DummyArgument::FromActual(
    std::string &&name, const Expr<SomeType> &expr, FoldingContext &context) {
  return std::visit(
      common::visitors{
          [&](const BOZLiteralConstant &) {
            return std::make_optional<DummyArgument>(std::move(name),
                DummyDataObject{
                    TypeAndShape{DynamicType::TypelessIntrinsicArgument()}});
          },
          [](const NullPointer &) { return std::optional<DummyArgument>{}; },
          [&](const ProcedureDesignator &designator) {
            if (auto proc{Procedure::Characterize(
                    designator, context.intrinsics())}) {
              return std::make_optional<DummyArgument>(
                  std::move(name), DummyProcedure{std::move(*proc)});
            } else {
              return std::optional<DummyArgument>{};
            }
          },
          [&](const ProcedureRef &call) {
            if (auto proc{
                    Procedure::Characterize(call, context.intrinsics())}) {
              return std::make_optional<DummyArgument>(
                  std::move(name), DummyProcedure{std::move(*proc)});
            } else {
              return std::optional<DummyArgument>{};
            }
          },
          [&](const auto &) {
            if (auto type{expr.GetType()}) {
              if (auto shape{GetShape(context, expr)}) {
                return std::make_optional<DummyArgument>(std::move(name),
                    DummyDataObject{TypeAndShape{*type, std::move(*shape)}});
              } else {
                return std::make_optional<DummyArgument>(
                    std::move(name), DummyDataObject{TypeAndShape{*type}});
              }
            } else {
              return std::optional<DummyArgument>{};
            }
          },
      },
      expr.u);
}

bool DummyArgument::IsOptional() const {
  return std::visit(
      common::visitors{
          [](const DummyDataObject &data) {
            return data.attrs.test(DummyDataObject::Attr::Optional);
          },
          [](const DummyProcedure &proc) {
            return proc.attrs.test(DummyProcedure::Attr::Optional);
          },
          [](const AlternateReturn &) { return false; },
      },
      u);
}

void DummyArgument::SetOptional(bool value) {
  std::visit(
      common::visitors{
          [value](DummyDataObject &data) {
            data.attrs.set(DummyDataObject::Attr::Optional, value);
          },
          [value](DummyProcedure &proc) {
            proc.attrs.set(DummyProcedure::Attr::Optional, value);
          },
          [](AlternateReturn &) { DIE("cannot set optional"); },
      },
      u);
}

bool DummyArgument::CanBePassedViaImplicitInterface() const {
  if (const auto *object{std::get_if<DummyDataObject>(&u)}) {
    return object->CanBePassedViaImplicitInterface();
  } else {
    return true;
  }
}

std::ostream &DummyArgument::Dump(std::ostream &o) const {
  if (!name.empty()) {
    o << name << '=';
  }
  if (pass) {
    o << " PASS";
  }
  std::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

FunctionResult::FunctionResult(DynamicType t) : u{TypeAndShape{t}} {}
FunctionResult::FunctionResult(TypeAndShape &&t) : u{std::move(t)} {}
FunctionResult::FunctionResult(Procedure &&p) : u{std::move(p)} {}
FunctionResult::~FunctionResult() {}

bool FunctionResult::operator==(const FunctionResult &that) const {
  return attrs == that.attrs && u == that.u;
}

std::optional<FunctionResult> FunctionResult::Characterize(
    const Symbol &symbol, const IntrinsicProcTable &intrinsics) {
  if (const auto *obj{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    if (auto type{TypeAndShape::Characterize(*obj)}) {
      FunctionResult result{std::move(*type)};
      CopyAttrs<FunctionResult, FunctionResult::Attr>(symbol, result,
          {
              {semantics::Attr::ALLOCATABLE, FunctionResult::Attr::Allocatable},
              {semantics::Attr::CONTIGUOUS, FunctionResult::Attr::Contiguous},
              {semantics::Attr::POINTER, FunctionResult::Attr::Pointer},
          });
      return result;
    }
  } else if (auto maybeProc{Procedure::Characterize(symbol, intrinsics)}) {
    FunctionResult result{std::move(*maybeProc)};
    result.attrs.set(FunctionResult::Attr::Pointer);
    return result;
  }
  return std::nullopt;
}

bool FunctionResult::IsAssumedLengthCharacter() const {
  if (const auto *ts{std::get_if<TypeAndShape>(&u)}) {
    return ts->type().IsAssumedLengthCharacter();
  } else {
    return false;
  }
}

bool FunctionResult::CanBeReturnedViaImplicitInterface() const {
  if (attrs.test(Attr::Pointer) || attrs.test(Attr::Allocatable)) {
    return false;  // 15.4.2.2(4)(b)
  } else if (const auto *typeAndShape{GetTypeAndShape()}) {
    if (typeAndShape->Rank() > 0) {
      return false;  // 15.4.2.2(4)(a)
    } else {
      const DynamicType &type{typeAndShape->type()};
      switch (type.category()) {
      case TypeCategory::Character:
        if (!type.IsAssumedLengthCharacter()) {
          if (const auto *param{type.charLength()}) {
            if (const auto &expr{param->GetExplicit()}) {
              return IsConstantExpr(*expr);  // 15.4.2.2(4)(c)
            }
          }
        }
        return false;
      case TypeCategory::Derived:
        if (!type.IsPolymorphic()) {
          const auto &spec{type.GetDerivedTypeSpec()};
          for (const auto &pair : spec.parameters()) {
            if (const auto &expr{pair.second.GetExplicit()}) {
              if (!IsConstantExpr(*expr)) {
                return false;  // 15.4.2.2(4)(c)
              }
            }
          }
          return true;
        }
        return false;
      default: return true;
      }
    }
  } else {
    return false;  // 15.4.2.2(4)(b) - procedure pointer
  }
}

std::ostream &FunctionResult::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  std::visit(
      common::visitors{
          [&](const TypeAndShape &ts) { ts.Dump(o); },
          [&](const CopyableIndirection<Procedure> &p) {
            p.value().Dump(o << " procedure(") << ')';
          },
      },
      u);
  return o;
}

Procedure::Procedure(FunctionResult &&fr, DummyArguments &&args, Attrs a)
  : functionResult{std::move(fr)}, dummyArguments{std::move(args)}, attrs{a} {}
Procedure::Procedure(DummyArguments &&args, Attrs a)
  : dummyArguments{std::move(args)}, attrs{a} {}
Procedure::~Procedure() {}

bool Procedure::operator==(const Procedure &that) const {
  return attrs == that.attrs && dummyArguments == that.dummyArguments &&
      functionResult == that.functionResult;
}

std::optional<Procedure> Procedure::Characterize(
    const semantics::Symbol &symbol, const IntrinsicProcTable &intrinsics) {
  Procedure result;
  CopyAttrs<Procedure, Procedure::Attr>(symbol, result,
      {
          {semantics::Attr::PURE, Procedure::Attr::Pure},
          {semantics::Attr::ELEMENTAL, Procedure::Attr::Elemental},
          {semantics::Attr::BIND_C, Procedure::Attr::BindC},
      });
  return std::visit(
      common::visitors{
          [&](const semantics::SubprogramDetails &subp)
              -> std::optional<Procedure> {
            if (subp.isFunction()) {
              if (auto fr{FunctionResult::Characterize(
                      subp.result(), intrinsics)}) {
                result.functionResult = std::move(fr);
              } else {
                return std::nullopt;
              }
            } else {
              result.attrs.set(Attr::Subroutine);
            }
            for (const semantics::Symbol *arg : subp.dummyArgs()) {
              if (!arg) {
                result.dummyArguments.emplace_back(AlternateReturn{});
              } else if (auto argCharacteristics{
                             DummyArgument::Characterize(*arg, intrinsics)}) {
                result.dummyArguments.emplace_back(
                    std::move(argCharacteristics.value()));
              } else {
                return std::nullopt;
              }
            }
            return result;
          },
          [&](const semantics::ProcEntityDetails &proc)
              -> std::optional<Procedure> {
            if (symbol.attrs().test(semantics::Attr::INTRINSIC)) {
              return intrinsics.IsUnrestrictedSpecificIntrinsicFunction(
                  symbol.name().ToString());
            }
            const semantics::ProcInterface &interface{proc.interface()};
            if (const semantics::Symbol * interfaceSymbol{interface.symbol()}) {
              return Characterize(*interfaceSymbol, intrinsics);
            } else {
              result.attrs.set(Attr::ImplicitInterface);
              const semantics::DeclTypeSpec *type{interface.type()};
              if (symbol.test(semantics::Symbol::Flag::Subroutine)) {
                // ignore any implicit typing
                result.attrs.set(Attr::Subroutine);
              } else if (type) {
                if (auto resultType{DynamicType::From(*type)}) {
                  result.functionResult = FunctionResult{*resultType};
                } else {
                  return std::nullopt;
                }
              } else if (symbol.test(semantics::Symbol::Flag::Function)) {
                return std::nullopt;
              }
              // The PASS name, if any, is not a characteristic.
              return result;
            }
          },
          [&](const semantics::ProcBindingDetails &binding) {
            if (auto result{Characterize(binding.symbol(), intrinsics)}) {
              if (const auto passIndex{binding.passIndex()}) {
                auto &passArg{result->dummyArguments.at(*passIndex)};
                passArg.pass = true;
                if (const auto passName{binding.passName()}) {
                  CHECK(passArg.name == passName->ToString());
                }
              }
              return result;
            } else {
              return std::optional<Procedure>{};
            }
          },
          [&](const semantics::UseDetails &use) {
            return Characterize(use.symbol(), intrinsics);
          },
          [&](const semantics::HostAssocDetails &assoc) {
            return Characterize(assoc.symbol(), intrinsics);
          },
          [](const auto &) { return std::optional<Procedure>{}; },
      },
      symbol.details());
}

std::optional<Procedure> Procedure::Characterize(
    const ProcedureDesignator &proc, const IntrinsicProcTable &intrinsics) {
  if (const auto *symbol{proc.GetSymbol()}) {
    if (auto result{characteristics::Procedure::Characterize(
            symbol->GetUltimate(), intrinsics)}) {
      return result;
    }
  } else if (const auto *intrinsic{proc.GetSpecificIntrinsic()}) {
    return intrinsic->characteristics.value();
  }
  return std::nullopt;
}

std::optional<Procedure> Procedure::Characterize(
    const ProcedureRef &ref, const IntrinsicProcTable &intrinsics) {
  if (auto callee{Characterize(ref.proc(), intrinsics)}) {
    if (callee->functionResult) {
      if (const Procedure *
          proc{callee->functionResult->IsProcedurePointer()}) {
        return {*proc};
      }
    }
  }
  return std::nullopt;
}

bool Procedure::CanBeCalledViaImplicitInterface() const {
  if (attrs.test(Attr::Elemental) || attrs.test(Attr::BindC)) {
    return false;  // 15.4.2.2(5,6)
  } else if (IsFunction() &&
      !functionResult->CanBeReturnedViaImplicitInterface()) {
    return false;
  } else {
    for (const DummyArgument &arg : dummyArguments) {
      if (!arg.CanBePassedViaImplicitInterface()) {
        return false;
      }
    }
    return true;
  }
}

std::ostream &Procedure::Dump(std::ostream &o) const {
  attrs.Dump(o, EnumToString);
  if (functionResult) {
    functionResult->Dump(o << "TYPE(") << ") FUNCTION";
  } else {
    o << "SUBROUTINE";
  }
  char sep{'('};
  for (const auto &dummy : dummyArguments) {
    dummy.Dump(o << sep);
    sep = ',';
  }
  return o << (sep == '(' ? "()" : ")");
}

// Utility class to determine if Procedures, etc. are distinguishable
class DistinguishUtils {
public:
  // Are these procedures distinguishable for a generic name?
  static bool Distinguishable(const Procedure &, const Procedure &);
  // Are these procedures distinguishable for a generic operator or assignment?
  static bool DistinguishableOpOrAssign(const Procedure &, const Procedure &);

private:
  struct CountDummyProcedures {
    CountDummyProcedures(const DummyArguments &args) {
      for (const DummyArgument &arg : args) {
        if (std::holds_alternative<DummyProcedure>(arg.u)) {
          total += 1;
          notOptional += !arg.IsOptional();
        }
      }
    }
    int total{0};
    int notOptional{0};
  };

  static bool Rule3Distinguishable(const Procedure &, const Procedure &);
  static const DummyArgument *Rule1DistinguishingArg(
      const DummyArguments &, const DummyArguments &);
  static int FindFirstToDistinguishByPosition(
      const DummyArguments &, const DummyArguments &);
  static int FindLastToDistinguishByName(
      const DummyArguments &, const DummyArguments &);
  static int CountCompatibleWith(const DummyArgument &, const DummyArguments &);
  static int CountNotDistinguishableFrom(
      const DummyArgument &, const DummyArguments &);
  static bool Distinguishable(const DummyArgument &, const DummyArgument &);
  static bool Distinguishable(const DummyDataObject &, const DummyDataObject &);
  static bool Distinguishable(const DummyProcedure &, const DummyProcedure &);
  static bool Distinguishable(const FunctionResult &, const FunctionResult &);
  static bool Distinguishable(const TypeAndShape &, const TypeAndShape &);
  static bool IsTkrCompatible(const DummyArgument &, const DummyArgument &);
  static bool IsTkrCompatible(const TypeAndShape &, const TypeAndShape &);
  static const DummyArgument *GetAtEffectivePosition(
      const DummyArguments &, int);
  static const DummyArgument *GetPassArg(const Procedure &);
};

// Simpler distinguishability rules for operators and assignment
bool DistinguishUtils::DistinguishableOpOrAssign(
    const Procedure &proc1, const Procedure &proc2) {
  auto &args1{proc1.dummyArguments};
  auto &args2{proc2.dummyArguments};
  if (args1.size() != args2.size()) {
    return true;  // C1511: distinguishable based on number of arguments
  }
  for (std::size_t i{0}; i < args1.size(); ++i) {
    if (Distinguishable(args1[i], args2[i])) {
      return true;  // C1511, C1512: distinguishable based on this arg
    }
  }
  return false;
}

bool DistinguishUtils::Distinguishable(
    const Procedure &proc1, const Procedure &proc2) {
  auto &args1{proc1.dummyArguments};
  auto &args2{proc2.dummyArguments};
  auto count1{CountDummyProcedures(args1)};
  auto count2{CountDummyProcedures(args2)};
  if (count1.notOptional > count2.total || count2.notOptional > count1.total) {
    return true;  // distinguishable based on C1514 rule 2
  }
  if (Rule3Distinguishable(proc1, proc2)) {
    return true;  // distinguishable based on C1514 rule 3
  }
  if (Rule1DistinguishingArg(args1, args2)) {
    return true;  // distinguishable based on C1514 rule 1
  }
  int pos1{FindFirstToDistinguishByPosition(args1, args2)};
  int name1{FindLastToDistinguishByName(args1, args2)};
  if (pos1 >= 0 && pos1 <= name1) {
    return true;  // distinguishable based on C1514 rule 4
  }
  int pos2{FindFirstToDistinguishByPosition(args2, args1)};
  int name2{FindLastToDistinguishByName(args2, args1)};
  if (pos2 >= 0 && pos2 <= name2) {
    return true;  // distinguishable based on C1514 rule 4
  }
  return false;
}

// C1514 rule 3: Procedures are distinguishable if both have a passed-object
// dummy argument and those are distinguishable.
bool DistinguishUtils::Rule3Distinguishable(
    const Procedure &proc1, const Procedure &proc2) {
  const DummyArgument *pass1{GetPassArg(proc1)};
  const DummyArgument *pass2{GetPassArg(proc2)};
  return pass1 && pass2 && Distinguishable(*pass1, *pass2);
}

// Find a non-passed-object dummy data object in one of the argument lists
// that satisfies C1514 rule 1. I.e. x such that:
// - m is the number of dummy data objects in one that are nonoptional,
//   are not passed-object, that x is TKR compatible with
// - n is the number of non-passed-object dummy data objects, in the other
//   that are not distinguishable from x
// - m is greater than n
const DummyArgument *DistinguishUtils::Rule1DistinguishingArg(
    const DummyArguments &args1, const DummyArguments &args2) {
  auto size1{args1.size()};
  auto size2{args2.size()};
  for (std::size_t i{0}; i < size1 + size2; ++i) {
    const DummyArgument &x{i < size1 ? args1[i] : args2[i - size1]};
    if (!x.pass && std::holds_alternative<DummyDataObject>(x.u)) {
      if (CountCompatibleWith(x, args1) >
              CountNotDistinguishableFrom(x, args2) ||
          CountCompatibleWith(x, args2) >
              CountNotDistinguishableFrom(x, args1)) {
        return &x;
      }
    }
  }
  return nullptr;
}

// Find the index of the first nonoptional non-passed-object dummy argument
// in args1 at an effective position such that either:
// - args2 has no dummy argument at that effective position
// - the dummy argument at that position is distinguishable from it
int DistinguishUtils::FindFirstToDistinguishByPosition(
    const DummyArguments &args1, const DummyArguments &args2) {
  int effective{0};  // position of arg1 in list, ignoring passed arg
  for (std::size_t i{0}; i < args1.size(); ++i) {
    const DummyArgument &arg1{args1.at(i)};
    if (!arg1.pass && !arg1.IsOptional()) {
      const DummyArgument *arg2{GetAtEffectivePosition(args2, effective)};
      if (!arg2 || Distinguishable(arg1, *arg2)) {
        return i;
      }
    }
    effective += !arg1.pass;
  }
  return -1;
}

// Find the index of the last nonoptional non-passed-object dummy argument
// in args1 whose name is such that either:
// - args2 has no dummy argument with that name
// - the dummy argument with that name is distinguishable from it
int DistinguishUtils::FindLastToDistinguishByName(
    const DummyArguments &args1, const DummyArguments &args2) {
  std::map<std::string, const DummyArgument *> nameToArg;
  for (const auto &arg2 : args2) {
    nameToArg.emplace(arg2.name, &arg2);
  }
  for (int i = args1.size() - 1; i >= 0; --i) {
    const DummyArgument &arg1{args1.at(i)};
    if (!arg1.pass && !arg1.IsOptional()) {
      auto it{nameToArg.find(arg1.name)};
      if (it == nameToArg.end() || Distinguishable(arg1, *it->second)) {
        return i;
      }
    }
  }
  return -1;
}

// Count the dummy data objects in args that are nonoptional, are not
// passed-object, and that x is TKR compatible with
int DistinguishUtils::CountCompatibleWith(
    const DummyArgument &x, const DummyArguments &args) {
  return std::count_if(args.begin(), args.end(), [&](const DummyArgument &y) {
    return !y.pass && !y.IsOptional() && IsTkrCompatible(x, y);
  });
}

// Return the number of dummy data objects in args that are not
// distinguishable from x and not passed-object.
int DistinguishUtils::CountNotDistinguishableFrom(
    const DummyArgument &x, const DummyArguments &args) {
  return std::count_if(args.begin(), args.end(), [&](const DummyArgument &y) {
    return !y.pass && std::holds_alternative<DummyDataObject>(y.u) &&
        !Distinguishable(y, x);
  });
}

bool DistinguishUtils::Distinguishable(
    const DummyArgument &x, const DummyArgument &y) {
  if (x.u.index() != y.u.index()) {
    return true;  // different kind: data/proc/alt-return
  }
  return std::visit(
      common::visitors{
          [&](const DummyDataObject &z) {
            return Distinguishable(z, std::get<DummyDataObject>(y.u));
          },
          [&](const DummyProcedure &z) {
            return Distinguishable(z, std::get<DummyProcedure>(y.u));
          },
          [&](const AlternateReturn &) { return false; },
      },
      x.u);
}

bool DistinguishUtils::Distinguishable(
    const DummyDataObject &x, const DummyDataObject &y) {
  using Attr = DummyDataObject::Attr;
  if (Distinguishable(x.type, y.type)) {
    return true;
  } else if (x.attrs.test(Attr::Allocatable) && y.attrs.test(Attr::Pointer) &&
      y.intent != common::Intent::In) {
    return true;
  } else if (y.attrs.test(Attr::Allocatable) && x.attrs.test(Attr::Pointer) &&
      x.intent != common::Intent::In) {
    return true;
  } else {
    return false;
  }
}

bool DistinguishUtils::Distinguishable(
    const DummyProcedure &x, const DummyProcedure &y) {
  const Procedure &xProc{x.procedure.value()};
  const Procedure &yProc{y.procedure.value()};
  if (Distinguishable(xProc, yProc)) {
    return true;
  } else {
    const std::optional<FunctionResult> &xResult{xProc.functionResult};
    const std::optional<FunctionResult> &yResult{yProc.functionResult};
    return xResult ? !yResult || Distinguishable(*xResult, *yResult)
                   : yResult.has_value();
  }
}

bool DistinguishUtils::Distinguishable(
    const FunctionResult &x, const FunctionResult &y) {
  if (x.u.index() != y.u.index()) {
    return true;  // one is data object, one is procedure
  }
  return std::visit(
      common::visitors{
          [&](const TypeAndShape &z) {
            return Distinguishable(z, std::get<TypeAndShape>(y.u));
          },
          [&](const CopyableIndirection<Procedure> &z) {
            return Distinguishable(z.value(),
                std::get<CopyableIndirection<Procedure>>(y.u).value());
          },
      },
      x.u);
}

bool DistinguishUtils::Distinguishable(
    const TypeAndShape &x, const TypeAndShape &y) {
  return !IsTkrCompatible(x, y) && !IsTkrCompatible(y, x);
}

// Compatibility based on type, kind, and rank
bool DistinguishUtils::IsTkrCompatible(
    const DummyArgument &x, const DummyArgument &y) {
  const auto *obj1{std::get_if<DummyDataObject>(&x.u)};
  const auto *obj2{std::get_if<DummyDataObject>(&y.u)};
  return obj1 && obj2 && IsTkrCompatible(obj1->type, obj2->type);
}
bool DistinguishUtils::IsTkrCompatible(
    const TypeAndShape &x, const TypeAndShape &y) {
  return x.type().IsTkCompatibleWith(y.type()) &&
      (x.attrs().test(TypeAndShape::Attr::AssumedRank) ||
          y.attrs().test(TypeAndShape::Attr::AssumedRank) ||
          x.Rank() == y.Rank());
}

// Return the argument at the given index, ignoring the passed arg
const DummyArgument *DistinguishUtils::GetAtEffectivePosition(
    const DummyArguments &args, int index) {
  for (const DummyArgument &arg : args) {
    if (!arg.pass) {
      if (index == 0) {
        return &arg;
      }
      --index;
    }
  }
  return nullptr;
}

// Return the passed-object dummy argument of this procedure, if any
const DummyArgument *DistinguishUtils::GetPassArg(const Procedure &proc) {
  for (const auto &arg : proc.dummyArguments) {
    if (arg.pass) {
      return &arg;
    }
  }
  return nullptr;
}

bool Distinguishable(const Procedure &x, const Procedure &y) {
  return DistinguishUtils::Distinguishable(x, y);
}

bool DistinguishableOpOrAssign(const Procedure &x, const Procedure &y) {
  return DistinguishUtils::DistinguishableOpOrAssign(x, y);
}

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyArgument)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyProcedure)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
}

template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;
