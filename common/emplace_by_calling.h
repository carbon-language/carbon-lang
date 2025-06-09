// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_EMPLACE_BY_CALLING_H_
#define CARBON_COMMON_EMPLACE_BY_CALLING_H_

#include <type_traits>
#include <utility>

namespace Carbon {

// A utility to use when calling an `emplace` function to emplace the result of
// a function call. Expected usage is:
//
//   my_widget_vec.emplace_back(EmplaceByCalling([&] {
//     return ConstructAWidget(...);
//   }));
//
// In this example, the result of `ConstructAWidget` will be constructed
// directly into the new element of `my_widget_vec`, without performing a copy
// or move.
//
// Note that the type of the argument to `emplace_back` is an `EmplaceByCalling`
// instance, not the type `DestT` stored in the container. When the `DestT`
// instance is eventually initialized directly from the `EmplaceByCalling`, a
// conversion function on `EmplaceByCalling` is used that converts to the type
// `DestT` being emplaced. This `DestT` initialization does not call an
// additional `DestT` copy or move constructor to initialize the result, and
// instead initializes it in-place in the container's storage, per the C++17
// guaranteed copy elision rules. Similarly, within the conversion function, the
// result is initialized directly by calling `make_fn`, again relying on
// guaranteed copy elision.
//
// Because the make function is called from the conversion function,
// `EmplaceByCalling` should only be used in contexts where it will be used to
// initialize a `DestT` object exactly once. This is generally true of `emplace`
// functions. Also, because the `make_fn` callback will be called after the
// container has made space for the new element, it should not inspect or modify
// the container that is being emplaced into.
template <typename MakeFnT>
class EmplaceByCalling {
 public:
  explicit(false) EmplaceByCalling(MakeFnT make_fn)
      : make_fn_(std::move(make_fn)) {}

  // Convert to the exact return type of the make function, by calling the make
  // function to construct the result. No implicit conversions are permitted
  // here, as that would mean we are not constructing the result in place.
  template <typename DestT>
    requires std::same_as<DestT, std::invoke_result_t<MakeFnT&&>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit(false) operator DestT() && {
    return std::move(make_fn_)();
  }

 private:
  MakeFnT make_fn_;
};

template <typename MakeFnT>
EmplaceByCalling(MakeFnT) -> EmplaceByCalling<MakeFnT>;

}  // namespace Carbon

#endif  // CARBON_COMMON_EMPLACE_BY_CALLING_H_
