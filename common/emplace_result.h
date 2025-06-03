// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_EMPLACE_RESULT_H_
#define CARBON_COMMON_EMPLACE_RESULT_H_

#include <type_traits>
#include <utility>

namespace Carbon {

// A utility to use when calling an `emplace` function to emplace the result of
// a function call. Expected usage is:
//
//   my_widget_vec.emplace_back(EmplaceResult([&] {
//     return ConstructAWidget(...);
//   }));
//
// In this example, the result of `ConstructAWidget` will be constructed
// directly into the new element of `my_widget_vec`, without performing a copy
// or move.
template <typename MakeFnT>
class EmplaceResult {
 public:
  explicit(false) EmplaceResult(MakeFnT make_fn)
      : make_fn_(std::move(make_fn)) {}

  // Convert to the exact return type of the make function, by calling the make
  // function to construct the result. No implicit conversions are permitted
  // here, as that would mean we are not constructing the result in place.
  template <typename DestT>
    requires std::same_as<DestT, std::invoke_result_t<MakeFnT>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit(false) operator DestT() && {
    return make_fn_();
  }

 private:
  MakeFnT make_fn_;
};

template <typename MakeFnT>
EmplaceResult(MakeFnT) -> EmplaceResult<MakeFnT>;

}  // namespace Carbon

#endif  // CARBON_COMMON_EMPLACE_RESULT_H_
