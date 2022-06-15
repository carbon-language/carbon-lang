//===-- lib/Evaluate/fold-reduction.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: DOT_PRODUCT, NORM2, PARITY

#ifndef FORTRAN_EVALUATE_FOLD_REDUCTION_H_
#define FORTRAN_EVALUATE_FOLD_REDUCTION_H_

#include "fold-implementation.h"

namespace Fortran::evaluate {

// Fold and validate a DIM= argument.  Returns false on error.
bool CheckReductionDIM(std::optional<int> &dim, FoldingContext &,
    ActualArguments &, std::optional<int> dimIndex, int rank);

// Fold and validate a MASK= argument.  Return null on error, absent MASK=, or
// non-constant MASK=.
Constant<LogicalResult> *GetReductionMASK(
    std::optional<ActualArgument> &maskArg, const ConstantSubscripts &shape,
    FoldingContext &);

// Common preprocessing for reduction transformational intrinsic function
// folding.  If the intrinsic can have DIM= &/or MASK= arguments, extract
// and check them.  If a MASK= is present, apply it to the array data and
// substitute identity values for elements corresponding to .FALSE. in
// the mask.  If the result is present, the intrinsic call can be folded.
template <typename T>
static std::optional<Constant<T>> ProcessReductionArgs(FoldingContext &context,
    ActualArguments &arg, std::optional<int> &dim, const Scalar<T> &identity,
    int arrayIndex, std::optional<int> dimIndex = std::nullopt,
    std::optional<int> maskIndex = std::nullopt) {
  if (arg.empty()) {
    return std::nullopt;
  }
  Constant<T> *folded{Folder<T>{context}.Folding(arg[arrayIndex])};
  if (!folded || folded->Rank() < 1) {
    return std::nullopt;
  }
  if (!CheckReductionDIM(dim, context, arg, dimIndex, folded->Rank())) {
    return std::nullopt;
  }
  if (maskIndex && static_cast<std::size_t>(*maskIndex) < arg.size() &&
      arg[*maskIndex]) {
    if (const Constant<LogicalResult> *mask{
            GetReductionMASK(arg[*maskIndex], folded->shape(), context)}) {
      // Apply the mask in place to the array
      std::size_t n{folded->size()};
      std::vector<typename Constant<T>::Element> elements;
      if (auto scalarMask{mask->GetScalarValue()}) {
        if (scalarMask->IsTrue()) {
          return Constant<T>{*folded};
        } else { // MASK=.FALSE.
          elements = std::vector<typename Constant<T>::Element>(n, identity);
        }
      } else { // mask is an array; test its elements
        elements = std::vector<typename Constant<T>::Element>(n, identity);
        ConstantSubscripts at{folded->lbounds()};
        for (std::size_t j{0}; j < n; ++j, folded->IncrementSubscripts(at)) {
          if (mask->values()[j].IsTrue()) {
            elements[j] = folded->At(at);
          }
        }
      }
      if constexpr (T::category == TypeCategory::Character) {
        return Constant<T>{static_cast<ConstantSubscript>(identity.size()),
            std::move(elements), ConstantSubscripts{folded->shape()}};
      } else {
        return Constant<T>{
            std::move(elements), ConstantSubscripts{folded->shape()}};
      }
    } else {
      return std::nullopt;
    }
  } else {
    return Constant<T>{*folded};
  }
}

// Generalized reduction to an array of one dimension fewer (w/ DIM=)
// or to a scalar (w/o DIM=).
template <typename T, typename ACCUMULATOR, typename ARRAY>
static Constant<T> DoReduction(const Constant<ARRAY> &array,
    std::optional<int> &dim, const Scalar<T> &identity,
    ACCUMULATOR &accumulator) {
  ConstantSubscripts at{array.lbounds()};
  std::vector<typename Constant<T>::Element> elements;
  ConstantSubscripts resultShape; // empty -> scalar
  if (dim) { // DIM= is present, so result is an array
    resultShape = array.shape();
    resultShape.erase(resultShape.begin() + (*dim - 1));
    ConstantSubscript dimExtent{array.shape().at(*dim - 1)};
    ConstantSubscript &dimAt{at[*dim - 1]};
    ConstantSubscript dimLbound{dimAt};
    for (auto n{GetSize(resultShape)}; n-- > 0;
         IncrementSubscripts(at, array.shape())) {
      dimAt = dimLbound;
      elements.push_back(identity);
      for (ConstantSubscript j{0}; j < dimExtent; ++j, ++dimAt) {
        accumulator(elements.back(), at);
      }
    }
  } else { // no DIM=, result is scalar
    elements.push_back(identity);
    for (auto n{array.size()}; n-- > 0;
         IncrementSubscripts(at, array.shape())) {
      accumulator(elements.back(), at);
    }
  }
  if constexpr (T::category == TypeCategory::Character) {
    return {static_cast<ConstantSubscript>(identity.size()),
        std::move(elements), std::move(resultShape)};
  } else {
    return {std::move(elements), std::move(resultShape)};
  }
}

// MAXVAL & MINVAL
template <typename T>
static Expr<T> FoldMaxvalMinval(FoldingContext &context, FunctionRef<T> &&ref,
    RelationalOperator opr, const Scalar<T> &identity) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Character);
  using Element = Scalar<T>;
  std::optional<int> dim;
  if (std::optional<Constant<T>> array{
          ProcessReductionArgs<T>(context, ref.arguments(), dim, identity,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    auto accumulator{[&](Element &element, const ConstantSubscripts &at) {
      Expr<LogicalResult> test{PackageRelation(opr,
          Expr<T>{Constant<T>{array->At(at)}}, Expr<T>{Constant<T>{element}})};
      auto folded{GetScalarConstantValue<LogicalResult>(
          test.Rewrite(context, std::move(test)))};
      CHECK(folded.has_value());
      if (folded->IsTrue()) {
        element = array->At(at);
      }
    }};
    return Expr<T>{DoReduction<T>(*array, dim, identity, accumulator)};
  }
  return Expr<T>{std::move(ref)};
}

// PRODUCT
template <typename T>
static Expr<T> FoldProduct(
    FoldingContext &context, FunctionRef<T> &&ref, Scalar<T> identity) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex);
  using Element = typename Constant<T>::Element;
  std::optional<int> dim;
  if (std::optional<Constant<T>> array{
          ProcessReductionArgs<T>(context, ref.arguments(), dim, identity,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    bool overflow{false};
    auto accumulator{[&](Element &element, const ConstantSubscripts &at) {
      if constexpr (T::category == TypeCategory::Integer) {
        auto prod{element.MultiplySigned(array->At(at))};
        overflow |= prod.SignedMultiplicationOverflowed();
        element = prod.lower;
      } else { // Real & Complex
        auto prod{element.Multiply(array->At(at))};
        overflow |= prod.flags.test(RealFlag::Overflow);
        element = prod.value;
      }
    }};
    if (overflow) {
      context.messages().Say(
          "PRODUCT() of %s data overflowed"_warn_en_US, T::AsFortran());
    } else {
      return Expr<T>{DoReduction<T>(*array, dim, identity, accumulator)};
    }
  }
  return Expr<T>{std::move(ref)};
}

// SUM
template <typename T>
static Expr<T> FoldSum(FoldingContext &context, FunctionRef<T> &&ref) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex);
  using Element = typename Constant<T>::Element;
  std::optional<int> dim;
  Element identity{}, correction{};
  if (std::optional<Constant<T>> array{
          ProcessReductionArgs<T>(context, ref.arguments(), dim, identity,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    bool overflow{false};
    auto accumulator{[&](Element &element, const ConstantSubscripts &at) {
      if constexpr (T::category == TypeCategory::Integer) {
        auto sum{element.AddSigned(array->At(at))};
        overflow |= sum.overflow;
        element = sum.value;
      } else { // Real & Complex: use Kahan summation
        auto next{array->At(at).Add(correction)};
        overflow |= next.flags.test(RealFlag::Overflow);
        auto sum{element.Add(next.value)};
        overflow |= sum.flags.test(RealFlag::Overflow);
        // correction = (sum - element) - next; algebraically zero
        correction =
            sum.value.Subtract(element).value.Subtract(next.value).value;
        element = sum.value;
      }
    }};
    if (overflow) {
      context.messages().Say(
          "SUM() of %s data overflowed"_warn_en_US, T::AsFortran());
    } else {
      return Expr<T>{DoReduction<T>(*array, dim, identity, accumulator)};
    }
  }
  return Expr<T>{std::move(ref)};
}

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_REDUCTION_H_
