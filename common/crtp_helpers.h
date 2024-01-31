// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_CRTP_HELPERS_H_
#define CARBON_COMMON_CRTP_HELPERS_H_

#include <compare>
#include <concepts>

namespace Carbon {

// Helper for building correct empty or stateless [CRTP] base classes.
//
// [CRTP]: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
//
// C++ conflates _compositional_ type authoring with inheritance. An effective
// pattern for doing compositional type design in C++ are CRTP base classes.
// However, because they are ultimately base classes, they have to be specially
// crafted in order to not disrupt useful C++ features such as defaulted
// comparisons.
//
// For an empty or stateless CRTP base class, we provide this helper that should
// itself follow the CRTP pattern when building a CRTP base class:
//
// ```cpp
// template <typename DerivedT>
// class ComposingWidget : EmptyCRTPBase<ComposingWidget<DerivedT>> {
//   ...
// };
// ```
//
// This will provide the necessary comparison functions to allow `DerivedT` to
// default its comparison functions (or not) as desired. These will be stateless
// comparison functions, and this helper is only valid when `ComposableWidget`
// is stateless and should not contribute in any way to the comparison function
// of `DerivedT`.
template <typename CRTPBaseT>
class EmptyCRTPBase {
  // For both equality and relational operators, we need to deduce the types on
  // both sides and require both to be exactly the provided `CRTPBaseT`. This
  // ensures these overloads are only used inside the implementation of a
  // defaulted operator in the derived type.
  //
  // The parameters are also accepted by value rather than by const reference
  // because this base is required to be empty. We don't want to force an
  // address to exist for it, and empty types have special low-cost
  // implementations in the ABI.
  template <typename LeftT, typename RightT>
    requires std::same_as<LeftT, CRTPBaseT> && std::same_as<RightT, CRTPBaseT>
  friend constexpr auto operator==(LeftT /*lhs*/, RightT /*rhs*/) -> bool {
    static_assert(std::is_empty_v<CRTPBaseT>,
                  "Can only use `EmptyCRTPBase` with empty and stateless CRTP "
                  "classes, as the operators it provides consider all "
                  "instances to be exactly equal to each other.");
    return true;
  }
  template <typename LeftT, typename RightT>
    requires std::same_as<LeftT, CRTPBaseT> && std::same_as<RightT, CRTPBaseT>
  friend constexpr auto operator<=>(LeftT /*lhs*/, RightT /*rhs*/) -> auto {
    static_assert(std::is_empty_v<CRTPBaseT>,
                  "Can only use `EmptyCRTPBase` with empty and stateless CRTP "
                  "classes, as the operators it provides consider all "
                  "instances to be exactly equal to each other.");
    return std::strong_ordering::equal;
  }
};

}  // namespace Carbon

#endif  // CARBON_COMMON_CRTP_HELPERS_H_
