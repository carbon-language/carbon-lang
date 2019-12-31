//===-- lib/evaluate/fold.cc ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#include "fold.h"
#include "fold-implementation.h"

namespace Fortran::evaluate {

template<typename T>
std::optional<Expr<T>> Folder<T>::GetNamedConstantValue(const Symbol &symbol0) {
  const Symbol &symbol{ResolveAssociations(symbol0).GetUltimate()};
  if (IsNamedConstant(symbol)) {
    if (const auto *object{
            symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      if (object->initWasValidated()) {
        const auto *constant{UnwrapConstantValue<T>(object->init())};
        CHECK(constant);
        return Expr<T>{*constant};
      }
      if (const auto &init{object->init()}) {
        if (auto dyType{DynamicType::From(symbol)}) {
          semantics::ObjectEntityDetails *mutableObject{
              const_cast<semantics::ObjectEntityDetails *>(object)};
          auto converted{
              ConvertToType(*dyType, std::move(mutableObject->init().value()))};
          // Reset expression now to prevent infinite loops if the init
          // expression depends on symbol itself.
          mutableObject->set_init(std::nullopt);
          if (converted) {
            *converted = Fold(context_, std::move(*converted));
            auto *unwrapped{UnwrapExpr<Expr<T>>(*converted)};
            CHECK(unwrapped);
            if (auto *constant{UnwrapConstantValue<T>(*unwrapped)}) {
              if (symbol.Rank() > 0) {
                if (constant->Rank() == 0) {
                  // scalar expansion
                  if (auto symShape{GetShape(context_, symbol)}) {
                    if (auto extents{AsConstantExtents(context_, *symShape)}) {
                      *constant = constant->Reshape(std::move(*extents));
                      CHECK(constant->Rank() == symbol.Rank());
                    }
                  }
                }
                if (constant->Rank() == symbol.Rank()) {
                  NamedEntity base{symbol};
                  if (auto lbounds{AsConstantExtents(
                          context_, GetLowerBounds(context_, base))}) {
                    constant->set_lbounds(*std::move(lbounds));
                  }
                }
              }
              mutableObject->set_init(AsGenericExpr(Expr<T>{*constant}));
              if (auto constShape{GetShape(context_, *constant)}) {
                if (auto symShape{GetShape(context_, symbol)}) {
                  if (CheckConformance(context_.messages(), *constShape,
                          *symShape, "initialization expression",
                          "PARAMETER")) {
                    mutableObject->set_initWasValidated();
                    return std::move(*unwrapped);
                  }
                } else {
                  context_.messages().Say(symbol.name(),
                      "Could not determine the shape of the PARAMETER"_err_en_US);
                }
              } else {
                context_.messages().Say(symbol.name(),
                    "Could not determine the shape of the initialization expression"_err_en_US);
              }
              mutableObject->set_init(std::nullopt);
            } else {
              std::stringstream ss;
              unwrapped->AsFortran(ss);
              context_.messages().Say(symbol.name(),
                  "Initialization expression for PARAMETER '%s' (%s) cannot be computed as a constant value"_err_en_US,
                  symbol.name(), ss.str());
            }
          } else {
            std::stringstream ss;
            init->AsFortran(ss);
            context_.messages().Say(symbol.name(),
                "Initialization expression for PARAMETER '%s' (%s) cannot be converted to its type (%s)"_err_en_US,
                symbol.name(), ss.str(), dyType->AsFortran());
          }
        }
      }
    }
  }
  return std::nullopt;
}

template<typename T>
std::optional<Constant<T>> Folder<T>::GetFoldedNamedConstantValue(
    const Symbol &symbol) {
  if (auto value{GetNamedConstantValue(symbol)}) {
    Expr<T> folded{Fold(context_, std::move(*value))};
    if (const Constant<T> *value{UnwrapConstantValue<T>(folded)}) {
      return *value;
    }
  }
  return std::nullopt;
}

static std::optional<Constant<SubscriptInteger>> GetConstantSubscript(
    FoldingContext &context, Subscript &ss, const NamedEntity &base, int dim) {
  ss = FoldOperation(context, std::move(ss));
  return std::visit(
      common::visitors{
          [](IndirectSubscriptIntegerExpr &expr)
              -> std::optional<Constant<SubscriptInteger>> {
            if (auto constant{
                    GetScalarConstantValue<SubscriptInteger>(expr.value())}) {
              return Constant<SubscriptInteger>{*constant};
            } else {
              return std::nullopt;
            }
          },
          [&](Triplet &triplet) -> std::optional<Constant<SubscriptInteger>> {
            auto lower{triplet.lower()}, upper{triplet.upper()};
            std::optional<ConstantSubscript> stride{ToInt64(triplet.stride())};
            if (!lower) {
              lower = GetLowerBound(context, base, dim);
            }
            if (!upper) {
              upper =
                  ComputeUpperBound(context, GetLowerBound(context, base, dim),
                      GetExtent(context, base, dim));
            }
            auto lbi{ToInt64(lower)}, ubi{ToInt64(upper)};
            if (lbi && ubi && stride && *stride != 0) {
              std::vector<SubscriptInteger::Scalar> values;
              while ((*stride > 0 && *lbi <= *ubi) ||
                  (*stride < 0 && *lbi >= *ubi)) {
                values.emplace_back(*lbi);
                *lbi += *stride;
              }
              return Constant<SubscriptInteger>{std::move(values),
                  ConstantSubscripts{
                      static_cast<ConstantSubscript>(values.size())}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

template<typename T>
std::optional<Constant<T>> Folder<T>::Folding(ArrayRef &aRef) {
  std::vector<Constant<SubscriptInteger>> subscripts;
  int dim{0};
  for (Subscript &ss : aRef.subscript()) {
    if (auto constant{GetConstantSubscript(context_, ss, aRef.base(), dim++)}) {
      subscripts.emplace_back(std::move(*constant));
    } else {
      return std::nullopt;
    }
  }
  if (Component * component{aRef.base().UnwrapComponent()}) {
    return GetConstantComponent(*component, &subscripts);
  } else if (std::optional<Constant<T>> array{
                 GetFoldedNamedConstantValue(aRef.base().GetLastSymbol())}) {
    return ApplySubscripts(*array, subscripts);
  } else {
    return std::nullopt;
  }
}

template<typename T>
std::optional<Constant<T>> Folder<T>::ApplySubscripts(const Constant<T> &array,
    const std::vector<Constant<SubscriptInteger>> &subscripts) {
  const auto &shape{array.shape()};
  const auto &lbounds{array.lbounds()};
  int rank{GetRank(shape)};
  CHECK(rank == static_cast<int>(subscripts.size()));
  std::size_t elements{1};
  ConstantSubscripts resultShape;
  ConstantSubscripts ssLB;
  for (const auto &ss : subscripts) {
    CHECK(ss.Rank() <= 1);
    if (ss.Rank() == 1) {
      resultShape.push_back(static_cast<ConstantSubscript>(ss.size()));
      elements *= ss.size();
      ssLB.push_back(ss.lbounds().front());
    }
  }
  ConstantSubscripts ssAt(rank, 0), at(rank, 0), tmp(1, 0);
  std::vector<Scalar<T>> values;
  while (elements-- > 0) {
    bool increment{true};
    int k{0};
    for (int j{0}; j < rank; ++j) {
      if (subscripts[j].Rank() == 0) {
        at[j] = subscripts[j].GetScalarValue().value().ToInt64();
      } else {
        CHECK(k < GetRank(resultShape));
        tmp[0] = ssLB[j] + ssAt[j];
        at[j] = subscripts[j].At(tmp).ToInt64();
        if (increment) {
          if (++ssAt[j] == resultShape[k]) {
            ssAt[j] = 0;
          } else {
            increment = false;
          }
        }
        ++k;
      }
      if (at[j] < lbounds[j] || at[j] >= lbounds[j] + shape[j]) {
        context_.messages().Say(
            "Subscript value (%jd) is out of range on dimension %d in reference to a constant array value"_err_en_US,
            static_cast<std::intmax_t>(at[j]), j + 1);
        return std::nullopt;
      }
    }
    values.emplace_back(array.At(at));
    CHECK(!increment || elements == 0);
    CHECK(k == GetRank(resultShape));
  }
  if constexpr (T::category == TypeCategory::Character) {
    return Constant<T>{array.LEN(), std::move(values), std::move(resultShape)};
  } else if constexpr (std::is_same_v<T, SomeDerived>) {
    return Constant<T>{array.result().derivedTypeSpec(), std::move(values),
        std::move(resultShape)};
  } else {
    return Constant<T>{std::move(values), std::move(resultShape)};
  }
}

template<typename T>
std::optional<Constant<T>> Folder<T>::ApplyComponent(
    Constant<SomeDerived> &&structures, const Symbol &component,
    const std::vector<Constant<SubscriptInteger>> *subscripts) {
  if (auto scalar{structures.GetScalarValue()}) {
    if (auto *expr{scalar->Find(component)}) {
      if (const Constant<T> *value{UnwrapConstantValue<T>(*expr)}) {
        if (!subscripts) {
          return std::move(*value);
        } else {
          return ApplySubscripts(*value, *subscripts);
        }
      }
    }
  } else {
    // A(:)%scalar_component & A(:)%array_component(subscripts)
    std::unique_ptr<ArrayConstructor<T>> array;
    if (structures.empty()) {
      return std::nullopt;
    }
    ConstantSubscripts at{structures.lbounds()};
    do {
      StructureConstructor scalar{structures.At(at)};
      if (auto *expr{scalar.Find(component)}) {
        if (const Constant<T> *value{UnwrapConstantValue<T>(*expr)}) {
          if (!array.get()) {
            // This technique ensures that character length or derived type
            // information is propagated to the array constructor.
            auto *typedExpr{UnwrapExpr<Expr<T>>(*expr)};
            CHECK(typedExpr);
            array = std::make_unique<ArrayConstructor<T>>(*typedExpr);
          }
          if (subscripts) {
            if (auto element{ApplySubscripts(*value, *subscripts)}) {
              CHECK(element->Rank() == 0);
              array->Push(Expr<T>{std::move(*element)});
            } else {
              return std::nullopt;
            }
          } else {
            CHECK(value->Rank() == 0);
            array->Push(Expr<T>{*value});
          }
        } else {
          return std::nullopt;
        }
      }
    } while (structures.IncrementSubscripts(at));
    // Fold the ArrayConstructor<> into a Constant<>.
    CHECK(array);
    Expr<T> result{Fold(context_, Expr<T>{std::move(*array)})};
    if (auto *constant{UnwrapConstantValue<T>(result)}) {
      return constant->Reshape(common::Clone(structures.shape()));
    }
  }
  return std::nullopt;
}

template<typename T>
std::optional<Constant<T>> Folder<T>::GetConstantComponent(Component &component,
    const std::vector<Constant<SubscriptInteger>> *subscripts) {
  if (std::optional<Constant<SomeDerived>> structures{std::visit(
          common::visitors{
              [&](const Symbol &symbol) {
                return Folder<SomeDerived>{context_}
                    .GetFoldedNamedConstantValue(symbol);
              },
              [&](ArrayRef &aRef) {
                return Folder<SomeDerived>{context_}.Folding(aRef);
              },
              [&](Component &base) {
                return Folder<SomeDerived>{context_}.GetConstantComponent(base);
              },
              [&](CoarrayRef &) {
                return std::optional<Constant<SomeDerived>>{};
              },
          },
          component.base().u)}) {
    return ApplyComponent(
        std::move(*structures), component.GetLastSymbol(), subscripts);
  } else {
    return std::nullopt;
  }
}

template<typename T> Expr<T> Folder<T>::Folding(Designator<T> &&designator) {
  if constexpr (T::category == TypeCategory::Character) {
    if (auto *substring{common::Unwrap<Substring>(designator.u)}) {
      if (std::optional<Expr<SomeCharacter>> folded{
              substring->Fold(context_)}) {
        if (auto value{GetScalarConstantValue<T>(*folded)}) {
          return Expr<T>{*value};
        }
      }
      if (auto length{ToInt64(Fold(context_, substring->LEN()))}) {
        if (*length == 0) {
          return Expr<T>{Constant<T>{Scalar<T>{}}};
        }
      }
    }
  }
  return std::visit(
      common::visitors{
          [&](SymbolRef &&symbol) {
            if (auto constant{GetFoldedNamedConstantValue(*symbol)}) {
              return Expr<T>{std::move(*constant)};
            }
            return Expr<T>{std::move(designator)};
          },
          [&](ArrayRef &&aRef) {
            aRef = FoldOperation(context_, std::move(aRef));
            if (auto c{Folding(aRef)}) {
              return Expr<T>{std::move(*c)};
            } else {
              return Expr<T>{Designator<T>{std::move(aRef)}};
            }
          },
          [&](Component &&component) {
            component = FoldOperation(context_, std::move(component));
            if (auto c{GetConstantComponent(component)}) {
              return Expr<T>{std::move(*c)};
            } else {
              return Expr<T>{Designator<T>{std::move(component)}};
            }
          },
          [&](auto &&x) {
            return Expr<T>{
                Designator<T>{FoldOperation(context_, std::move(x))}};
          },
      },
      std::move(designator.u));
}

FOR_EACH_SPECIFIC_TYPE(template class Folder, )

Expr<SomeDerived> FoldOperation(
    FoldingContext &context, StructureConstructor &&structure) {
  StructureConstructor result{structure.derivedTypeSpec()};
  for (auto &&[symbol, value] : std::move(structure)) {
    result.Add(symbol, Fold(context, std::move(value.value())));
  }
  return Expr<SomeDerived>{Constant<SomeDerived>{std::move(result)}};
}

Component FoldOperation(FoldingContext &context, Component &&component) {
  return {FoldOperation(context, std::move(component.base())),
      component.GetLastSymbol()};
}

NamedEntity FoldOperation(FoldingContext &context, NamedEntity &&x) {
  if (Component * c{x.UnwrapComponent()}) {
    return NamedEntity{FoldOperation(context, std::move(*c))};
  } else {
    return std::move(x);
  }
}

Triplet FoldOperation(FoldingContext &context, Triplet &&triplet) {
  MaybeExtentExpr lower{triplet.lower()};
  MaybeExtentExpr upper{triplet.upper()};
  return {Fold(context, std::move(lower)), Fold(context, std::move(upper)),
      Fold(context, triplet.stride())};
}

Subscript FoldOperation(FoldingContext &context, Subscript &&subscript) {
  return std::visit(
      common::visitors{
          [&](IndirectSubscriptIntegerExpr &&expr) {
            expr.value() = Fold(context, std::move(expr.value()));
            return Subscript(std::move(expr));
          },
          [&](Triplet &&triplet) {
            return Subscript(FoldOperation(context, std::move(triplet)));
          },
      },
      std::move(subscript.u));
}

ArrayRef FoldOperation(FoldingContext &context, ArrayRef &&arrayRef) {
  NamedEntity base{FoldOperation(context, std::move(arrayRef.base()))};
  for (Subscript &subscript : arrayRef.subscript()) {
    subscript = FoldOperation(context, std::move(subscript));
  }
  return ArrayRef{std::move(base), std::move(arrayRef.subscript())};
}

CoarrayRef FoldOperation(FoldingContext &context, CoarrayRef &&coarrayRef) {
  std::vector<Subscript> subscript;
  for (Subscript x : coarrayRef.subscript()) {
    subscript.emplace_back(FoldOperation(context, std::move(x)));
  }
  std::vector<Expr<SubscriptInteger>> cosubscript;
  for (Expr<SubscriptInteger> x : coarrayRef.cosubscript()) {
    cosubscript.emplace_back(Fold(context, std::move(x)));
  }
  CoarrayRef folded{std::move(coarrayRef.base()), std::move(subscript),
      std::move(cosubscript)};
  if (std::optional<Expr<SomeInteger>> stat{coarrayRef.stat()}) {
    folded.set_stat(Fold(context, std::move(*stat)));
  }
  if (std::optional<Expr<SomeInteger>> team{coarrayRef.team()}) {
    folded.set_team(
        Fold(context, std::move(*team)), coarrayRef.teamIsTeamNumber());
  }
  return folded;
}

DataRef FoldOperation(FoldingContext &context, DataRef &&dataRef) {
  return std::visit(
      common::visitors{
          [&](SymbolRef symbol) { return DataRef{*symbol}; },
          [&](auto &&x) {
            return DataRef{FoldOperation(context, std::move(x))};
          },
      },
      std::move(dataRef.u));
}

Substring FoldOperation(FoldingContext &context, Substring &&substring) {
  auto lower{Fold(context, substring.lower())};
  auto upper{Fold(context, substring.upper())};
  if (const DataRef * dataRef{substring.GetParentIf<DataRef>()}) {
    return Substring{FoldOperation(context, DataRef{*dataRef}),
        std::move(lower), std::move(upper)};
  } else {
    auto p{*substring.GetParentIf<StaticDataObject::Pointer>()};
    return Substring{std::move(p), std::move(lower), std::move(upper)};
  }
}

ComplexPart FoldOperation(FoldingContext &context, ComplexPart &&complexPart) {
  DataRef complex{complexPart.complex()};
  return ComplexPart{
      FoldOperation(context, std::move(complex)), complexPart.part()};
}

std::optional<std::int64_t> GetInt64Arg(
    const std::optional<ActualArgument> &arg) {
  if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

std::optional<std::int64_t> GetInt64ArgOr(
    const std::optional<ActualArgument> &arg, std::int64_t defaultValue) {
  if (!arg) {
    return defaultValue;
  } else if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&iDo) {
  if (std::optional<ConstantSubscript> value{context.GetImpliedDo(iDo.name)}) {
    return Expr<ImpliedDoIndex::Result>{*value};
  } else {
    return Expr<ImpliedDoIndex::Result>{std::move(iDo)};
  }
}

template class ExpressionBase<SomeDerived>;
template class ExpressionBase<SomeType>;

}
