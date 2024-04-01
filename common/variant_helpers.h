// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_VARIANT_HELPERS_H_
#define CARBON_COMMON_VARIANT_HELPERS_H_

#include <variant>

#include "common/error.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

namespace Internal {

// Form an overload set from a list of functions. For example:
//
// ```
// auto overloaded = Overload{[] (int) {}, [] (float) {}};
// ```
template <typename... Fs>
struct Overload : Fs... {
  using Fs::operator()...;
};
template <typename... Fs>
Overload(Fs...) -> Overload<Fs...>;

}  // namespace Internal

// Pattern-match against the type of the value stored in the variant `V`. Each
// element of `fs` should be a function that takes one or more of the variant
// values in `V`.
template <typename V, typename... Fs>
auto VariantMatch(V&& v, Fs&&... fs) -> decltype(auto) {
  return std::visit(Internal::Overload{std::forward<Fs&&>(fs)...},
                    std::forward<V&&>(v));
}

}  // namespace Carbon

#endif  // CARBON_COMMON_VARIANT_HELPERS_H_
